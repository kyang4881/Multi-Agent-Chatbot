import gradio as gr
import plotly.graph_objects as go
from FacilitiesMaintenanceDAO import FacilitiesMaintenanceDAO
from VolumeByBusStopDAO import VolumeByBusStopDAO
from VolumeByTrainStationDAO import VolumeByTrainStationDAO
from RoadWorksDAO import RoadWorksDAO
from TrafficFlowDAO import TrafficFlowDAO
from sustainability import sustainability
from LLMOpenAIInterface import LLMOpenAI
from LLMSQLInterface import LLMSQLInterface
from bigq_extract import query_db
from openai import OpenAI
import json

config=None
with open (".config.json",'r') as f:
    config=json.load(f)


# Extract data from BigQuery
query_data = query_db()

## Tab 1
showContent=[]
hideContent=[]
for i in range(6):
    showContent.append(None)
    hideContent.append(None)
guideText=''.join(open("./sample_assets/guideText.txt",mode="r").readlines())


## chart 0: traffic incidents barplot
def currentFunction():
    return(gr.Group(visible=True),gr.Column(visible=False),gr.Button(visible=True))
showContent[0]=currentFunction

def currentFunction():
    return(gr.Group(visible=False),gr.Column(visible=True),gr.Button(visible=False))
hideContent[0]=currentFunction

query = "SELECT * FROM `new-chatbot-428815.Traffic.traffic_incidents`"
traffic_incidents_df = query_data.run_query(query)

trafficIncidents=traffic_incidents_df.groupby("Type").size().reset_index(name='counts')


## chart 1: road works
roadWorksDAO=RoadWorksDAO()
roadWorksDAO.loadLocalData()

def currentFunction():
    return(gr.Group(visible=True),gr.Column(visible=False),gr.Button(visible=True))
showContent[1]=currentFunction

def currentFunction():
    return(gr.Group(visible=False),gr.Column(visible=True),gr.Button(visible=False))
hideContent[1]=currentFunction

def filterByRoadName(selectedRoadName):
    return(
        gr.Dataframe(
            value=roadWorksDAO.getDataframeByRoadName(selectedRoadName),
            show_label=False,
            wrap=True
        ))

## chart 2: Facilities Maintenance
facilitiesMaintenanceDAO=FacilitiesMaintenanceDAO()
facilitiesMaintenanceDAO.loadFromCSV()

def currentFunction():
    return(gr.Group(visible=True),gr.Column(visible=False),gr.Button(visible=True))
showContent[2]=currentFunction

def currentFunction():
    return(gr.Group(visible=False),gr.Column(visible=True),gr.Button(visible=False))
hideContent[2]=currentFunction

def filterByStationName(selectedStationName):
    return(
        gr.Dataframe(
            value=facilitiesMaintenanceDAO.getDataframeByStationName(selectedStationName),
            show_label=False,
            wrap=True
        ))


## chart 3: Passenger volume by Bus Stop
def currentFunction():
    return(gr.Group(visible=True),gr.Column(visible=False),gr.Button(visible=True))
showContent[3]=currentFunction

def currentFunction():
    return(gr.Group(visible=False),gr.Column(visible=True),gr.Button(visible=False))
hideContent[3]=currentFunction

volumeByBusStopDAO=VolumeByBusStopDAO()
volumeByBusStopDAO.loadLocalData()


## chart 4: Passenger volume by Train Station
def currentFunction():
    return(gr.Group(visible=True),gr.Column(visible=False),gr.Button(visible=True))
showContent[4]=currentFunction

def currentFunction():
    return(gr.Group(visible=False),gr.Column(visible=True),gr.Button(visible=False))
hideContent[4]=currentFunction

volumeByTrainStationDAO=VolumeByTrainStationDAO()
volumeByTrainStationDAO.loadLocalData()


## Tab 2

trafficFlowDAO=TrafficFlowDAO()
trafficFlowDAO.loadFromLocal()

def filter_map(date='01/11/2023 7:00'):
    if date=="__Most Recent__":
        thisDate='01/11/2023 7:00'
    else:
        thisDate=date
    densitymapbox=trafficFlowDAO.getScattermapbox(thisDate)
    fig = go.Figure(densitymapbox)

    fig.update_layout(
        autosize=True,
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=1.368,
                lon=103.794
            ),
            pitch=0,
            zoom=10
        ),
    )

    return fig




def changeQuickQueryToFrom(choice):
    if choice=="FROM":
        return(
                gr.Dropdown(choices=["FROM","TO"],value="TO",interactive=False,show_label=False,scale=1,min_width=30,filterable=False)
            )
    
    return(
        gr.Dropdown(choices=["FROM","TO"],value="FROM",interactive=False,show_label=False,scale=1,min_width=30,filterable=False)
    )

def emergencyReportGenerator(text):
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": 
            """
                You are an emergency response advice provider, and your clients are traffic authorities.\n
                You provide emergency response suggestions.\n
                If your answer contains MRT Station Name, provide its station code.\n
                If your answer contains MRT Line Name, provide its line code.\n
                If your answer contains bus stop name, provide its code.\n

                You will predict at most 3 affected stations and/or stops. Provide number of passengers affected per hour for each station and/or stops you mentioned.
                When predicting affected stations and/or stops, consider not only where the emergency occurred, but also the route-adjacent stations and/or stops.\n
                Predict the types of emergency services the user needs to deploy and to which location in a separate paragraph. The emergency service types must strictly be: hospital, fire station, police station. However,if you think user don't need any of these services, omit this part.\n
                Your answer should be as concise and professional as possible and should not contain any non-specific advice.\n
                You don't need to explain your rationale.\n
                No approximate values are allowed in your answer. If you cannot provide an exact number, provide a range bounded by the exact number.\n
            """
            },
            {"role": "user", "content": text}
        ]
    )
    returnText=completion.choices[0].message.content

    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": 
            """
                You are a data analyst serving for Singaporean transport system. You calculate and summarize data\n
                The input is a emergency response report including several hourly numbers of passenger affected in several train stations and/or bus stops respectively.\n
                You will devide those numbers by 102, then match each devided number with corresponding station or stop name.\n
                You will list those devided numbers in following format: 1. name1 need devidedNumber1 extra bus trips per hour\n
                You may not show calculations.
            """
            },
            {"role": "user", "content": returnText}
        ]
    )

    returnText=returnText+"\n\nBus Deployment Recommendation:\n"+completion.choices[0].message.content

    return(returnText)

def reportReader(reportText):
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": 
            """
                You are a data preparation system. You serve Singaporean transport system\n
                Your input will be a traffic emergency response report.\n
                You will analyze which emergency services are suggested to be deployed to which destinations in this report\n
                You should treat bus interchange as an emergency service if you notice the report mentions that there are passengers affected somewhere. However, when you are listing the results, you should put bus interchange later in the list.\n
                If you think there is no emergency service included in this report, you need to strictly return:[[]]\n
                Otherwise, your answer must strictly be in following format:[["emergencyServiceName1","destination1"],["emergencyServiceName2","destination2"],...]
            """
            },
            {"role": "user", "content": reportText}
        ]
    )

    try:
        returnList=json.loads(completion.choices[0].message.content)
        if len(returnList[0])==0:
            returnList=None
    except:
        returnList=None

    return(returnList)

def intentionClassifier(text):
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": 
            """
                You are an intention classifier.\n
                Your task is to identify the type of intention of the query.\n
                There are 3 possible types of intention: Describing Traffic Event, Requesting a Route, None of Above.\n
                You must answer strictly according to the following rules:\n
                If you think the intention of the query is Describing Traffic Event, you will return 1\n
                If you think the intention of the query is Requesting a Route, you will return 2\n
                If you think the intention of the query is None of Above, you will return 0\n
                You must directly return the number without anything else.
            """
            },
            {"role": "user", "content": text}
        ]
    )
    try:
        typeNumber=int(str(completion.choices[0].message.content))
        if typeNumber not in [1,2]:
            return(0)
        return(typeNumber)
    except:
        return(0)

def openChatter(text):
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": 
            """
                You are a Singaporean Traffic Assistant.\n
                Your user is a traffic planner, you will give suggestions from this perspective.
            """
            },
            {"role": "user", "content": text}
        ]
    )
    return(str(completion.choices[0].message.content))

def busInterchangeQuery(text):
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": 
            """
                You are a Singaporean Bus Interchange Searching Service.\n
                You will output strictly according to the following rules:\n
                If you identify that input text contains words "bus interchange near A", you will search for a bus interchange near A and replace those words in the text. Then you will return the modified text without anything else.\n
                If you don't identify that input text contains words "bus interchange near A", you will directly return the input text without anything else.
            """
            },
            {"role": "user", "content": text}
        ]
    )
    return(completion.choices[0].message.content)


def routeQuery(text):
    modifiedText=busInterchangeQuery(text)
    print("modified:"+modifiedText)

    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": 
            """
                You are a Navigation Assistant that is familiar with the Singapore map.\n
                User will query you for a route. You will describe the route by specifying start place name and end place name.\n
                User may use vague place names, including hospitals, fire stations, police stations. You need to search for specific place names to replace the vague place names.\n
                Your response must strictly be in following format:[StartPlaceName#EndPlaceName]\n
            """
            },
            {"role": "user", "content": modifiedText}
        ]
    )
    # 
    returnText=str(completion.choices[0].message.content)[1:-1]
    returns=returnText.split("#")

    returns[0]=returns[0].replace(" ","+")
    returns[1]=returns[1].replace(" ","+")

    returns[0]="[\""+returns[0]+",+Singapore\","
    returns[1]="\""+returns[1]+",+Singapore\"]"
    returnText=returns[0]+returns[1]

    print(returnText)

    return returnText

def generateHTMLText(routeQueryReturnText):
    try:
        if routeQueryReturnText!=None:
            return("<iframe src=?body="+routeQueryReturnText+" width=100%% height=560px></iframe>")
        else:
            return("<iframe src= width=100%% height=560px></iframe>")
            
    except:
        return("<iframe src= width=100%% height=560px></iframe>")

def submitFreeQueryTextbox(history,text):
    intentionTypeNumber=intentionClassifier(text)
    print(intentionTypeNumber)
    returnHtmlBox=None
    reportText=None
    dropDown=gr.Dropdown(choices=None,type='index',value=None,interactive=False,allow_custom_value=False)
    freeQueryReportReaderReturnList=None
    jsonText=None

    if intentionTypeNumber==0:
        reportText="""
            ***General Chat Mode***\n\
            ---Tips: If you wish to enter emergency report mode, you need to describe exactly what happened at where.---\n\n
        """

        reportText=reportText+openChatter(text)
        returnHtmlBox=gr.HTML(generateHTMLText(None))

    elif intentionTypeNumber==1:
        reportText="***Emergency Report Mode***\n\n"
        reportText+=emergencyReportGenerator(text)
        freeQueryReportReaderReturnList=reportReader(reportText)
        print(freeQueryReportReaderReturnList)
        if freeQueryReportReaderReturnList!=None:
            choises=[item[0]+" TO "+item[1] for item in freeQueryReportReaderReturnList]
            dropDown=gr.Dropdown(choices=choises,type='index',value=choises[0],interactive=True,allow_custom_value=False)
            returnHtmlBox=submitQuickQueryTextbox("FROM",freeQueryReportReaderReturnList[0][0],"TO",freeQueryReportReaderReturnList[0][1])
        else:
            returnHtmlBox=gr.HTML(generateHTMLText(None))
    elif intentionTypeNumber==2:
        reportText="***Route Mode***\n\n"
        reportText="Got it."
        returnHtmlBox=gr.HTML(generateHTMLText(routeQuery(text)))

    if freeQueryReportReaderReturnList!=None:
        jsonText=json.dumps(freeQueryReportReaderReturnList)

    returnList=[
        history+[
            [
                text,
                reportText
            ]
        ],
        returnHtmlBox,
        dropDown,
        jsonText
    ]

    return(returnList)

def changeFreeQueryDropDown(index,jsonText):
    print("debug index:")
    print(index)
    freeQueryReportReaderReturnList=json.loads(jsonText)

    return(submitQuickQueryTextbox("FROM",freeQueryReportReaderReturnList[index][0],"TO",freeQueryReportReaderReturnList[index][1]))

def submitQuickQueryTextbox(quickQueryFromTo,quickQueryDorpDown,quickQueryToFrom,quickQueryTextbox):
    text="Show me a route "+quickQueryFromTo+" a "+quickQueryDorpDown+" near "+quickQueryTextbox+" "+quickQueryToFrom+" "+quickQueryTextbox
    print(text)
    return(gr.HTML(generateHTMLText(routeQuery(text))))

def loadURLParams(request:gr.Request):
    urlParamsDict=dict(request.query_params)
    print(urlParamsDict)
    selected=None
    jsonString=None
    returnFreeQueryTextbox=None

    if 'tab' in urlParamsDict:
        selected=urlParamsDict['tab']
        del urlParamsDict['tab']

    if 'query' in urlParamsDict:
        returnFreeQueryTextbox=gr.Textbox(value=urlParamsDict['query'])
        del urlParamsDict["query"]

    if len(urlParamsDict)>0:
        jsonString=json.dumps(urlParamsDict)

    return([gr.Textbox(value=jsonString),gr.Tabs(selected=selected),returnFreeQueryTextbox])

def defaultFreeQueryByURLParams(globalJsonBuffer):
    globalDict=None
    try:
        globalDict=json.loads(globalJsonBuffer)
    except:
        return([None,None])
    print(globalDict)

    if 'query' in globalDict:
        freeQueryText=globalDict['query']
        del globalDict['query']

        return(
            [
                gr.Textbox(value=freeQueryText),
                gr.Textbox(value=json.dumps(globalDict))
            ]
        )
    
    return([None,None])


# GUI 

db_llm = LLMSQLInterface()
doc_llm = LLMOpenAI()
sustainability_forecast = sustainability()

try:
    demo.close()
except:
    pass

with gr.Blocks(title="BooleanPirates") as demo:
    globalJsonBuffer=gr.Textbox(value=None,visible=False)
    with gr.Row(variant='panel'):
        with gr.Column():
            gr.Tab("TransportGPT",interactive=False)
            with gr.Column():          
                with gr.Tab("Query Database"):
                    with gr.Group():
                        chatbot=gr.Chatbot(show_label=True)
                        msg=gr.Textbox(placeholder="Press enter to submit your query")
                        clear=gr.ClearButton([msg,chatbot],value="Clear")
                        msg.submit(db_llm.create_conversation,[msg,chatbot],[msg,chatbot])

                with gr.Tab("Query Document"):
                    with gr.Column("Document Upload", elem_classes=["upload_container"]):
                        gr.Interface(
                            fn=doc_llm.process_file,
                            inputs=["file"],
                            outputs="text",
                            title="File Uploader",
                            description="Upload a file to start document search (Supported file types: DOCX, TXT, PDF, CSV)",
                            allow_flagging="auto"
                        )
                    with gr.Column("Chatbot 2", elem_classes=["chatbot2_container"]):
                        chatbot = gr.Chatbot(show_label=True)
                        msg=gr.Textbox(placeholder="Press enter to submit your query")
                        clear = gr.ClearButton([msg, chatbot])
                        msg.submit(doc_llm.create_conversation, [msg, chatbot], [msg, chatbot])
            
        with gr.Column(min_width=950):
            gr.Tab("TransportViz",interactive=False)
            with gr.Column():
                with gr.Tab("Charts"):
                    with gr.Column() as dashboard:
                        with gr.Column() as dashboardSelector:
                            with gr.Row():
                                buttons=[
                                    gr.Button(
                                        value="Traffic Incidents",
                                        scale=4,
                                        ),
                                    gr.Button(
                                        value="Road Works",
                                        scale=4,
                                        ),
                                    gr.Button(
                                        value="Train Station Facility Maintenance",
                                        scale=4,
                                        ),
                                    gr.Button(
                                        value="Bus Stop Passenger Volume",
                                        scale=4,
                                        ),
                                    gr.Button(
                                        value="Train Station Passenger Volume",
                                        scale=4,
                                        )
                                ]
                            defaultGuide=gr.Textbox(value=guideText,label="")

                        with gr.Group() as container:
                            returnButtons=[
                                gr.Button(value="Back",visible=False),
                                gr.Button(value="Back",visible=False),
                                gr.Button(value="Back",visible=False),
                                gr.Button(value="Back",visible=False),
                                gr.Button(value="Back",visible=False)
                            ]
                            contents=[
                                gr.Group(
                                    visible=False
                                    ),
                                gr.Group(
                                    visible=False,
                                    ),
                                gr.Group(
                                    visible=False,
                                    ),
                                gr.Group(
                                    visible=False,
                                    ),
                                gr.Group(
                                    visible=False,
                                    )
                            ]
                            with contents[0]:
                                gr.Label(value="Traffic Incidents",show_label=False)
                                gr.BarPlot(
                                    show_label=False,
                                    value=trafficIncidents,
                                    x="Type",
                                    y="counts",
                                    vertical=False,
                                    tooltip=["counts"],
                                    height=360,
                                    width=720,
                                    color="Type"
                                )

                            with contents[1]:
                                gr.Label(value="Road Works",show_label=False)
                                gr.Label(value="Currently there's "+str(roadWorksDAO.roadWorks.shape[0])+" road work items existing",show_label=False)
                                roadWorksDropDown=gr.Dropdown(["__ALL__"]+roadWorksDAO.getUniqueRoadNameList(),label="Type here then select to filter by road name")
                                roadWorksDataframe=gr.Dataframe(
                                    value=roadWorksDAO.roadWorks,
                                    show_label=False,
                                    wrap=True
                                )

                            roadWorksDropDown.change(filterByRoadName,roadWorksDropDown,roadWorksDataframe)

                            with contents[2]:
                                gr.Label(value="Train Station Facility Maintenance",show_label=False)
                                gr.Label(value="There's "+str(len(facilitiesMaintenanceDAO.getUniqueStationNameList()))+" stations with maintenance item",show_label=False)
                                facilitiesMaintenanceDropDown=gr.Dropdown(["__ALL__"]+facilitiesMaintenanceDAO.getUniqueStationNameList(),label="Type here then select to filter by station name")
                                facilitiesMaintenanceDataframe=gr.Dataframe(
                                    value=facilitiesMaintenanceDAO.pdTable,
                                    show_label=False,
                                    wrap=True
                                )
                            
                            facilitiesMaintenanceDropDown.change(filterByStationName,facilitiesMaintenanceDropDown,facilitiesMaintenanceDataframe)

                            with contents[3]:
                                gr.Label(value="Bus Stop Passenger Volume",show_label=False)
                                with gr.Row():
                                    chart3_1=gr.BarPlot(
                                        label="Boarding",
                                        value=volumeByBusStopDAO.volumeByBusStop.sort_values(by="TOTAL_TAP_IN_VOLUME",ascending=False).head(10),
                                        x="Description",
                                        x_title="Bus Stop",
                                        x_label_angle=45,
                                        y="TOTAL_TAP_IN_VOLUME",
                                        y_title="Boarding Volume",
                                        tooltip=["PT_CODE","RoadName","Description","TOTAL_TAP_IN_VOLUME"],
                                        height=360,
                                        width=475
                                        )
                                    chart3_2=gr.BarPlot(
                                        label="Alighting",
                                        value=volumeByBusStopDAO.volumeByBusStop.sort_values(by="TOTAL_TAP_OUT_VOLUME",ascending=False).head(10),
                                        x="Description",
                                        x_title="Bus Stop",
                                        x_label_angle=45,
                                        y="TOTAL_TAP_OUT_VOLUME",
                                        y_title="Alighting Volume",
                                        tooltip=["PT_CODE","RoadName","Description","TOTAL_TAP_OUT_VOLUME"],
                                        height=360,
                                        width=475
                                        )
                            
                            with contents[4]:
                                gr.Label(value="Train Station Passenger Volume",show_label=False)
                                with gr.Row():
                                    chart4_1=gr.BarPlot(
                                        label="Boarding",
                                        value=volumeByTrainStationDAO.volumeByTrainStation.sort_values(by="TOTAL_TAP_IN_VOLUME",ascending=False).head(10),
                                        x="mrt_station_english",
                                        x_title="Train Station",
                                        x_label_angle=45,
                                        y="TOTAL_TAP_IN_VOLUME",
                                        y_title="Boarding Volume",
                                        tooltip=["PT_CODE","mrt_station_english","mrt_line_english","TOTAL_TAP_IN_VOLUME"],
                                        height=360,
                                        width=475
                                        )
                                    chart4_2=gr.BarPlot(
                                        label="Alighting",
                                        value=volumeByTrainStationDAO.volumeByTrainStation.sort_values(by="TOTAL_TAP_OUT_VOLUME",ascending=False).head(10),
                                        x="mrt_station_english",
                                        x_title="Train Station",
                                        x_label_angle=45,
                                        y="TOTAL_TAP_OUT_VOLUME",
                                        y_title="Alighting Volume",
                                        tooltip=["PT_CODE","mrt_station_english","mrt_line_english","TOTAL_TAP_OUT_VOLUME"],
                                        height=360,
                                        width=475
                                        )
                                                            
                        for i in range(5):
                            buttons[i].click(fn=showContent[i],outputs=[contents[i],dashboardSelector,returnButtons[i]])
                            returnButtons[i].click(fn=hideContent[i],outputs=[contents[i],dashboardSelector,returnButtons[i]])

                with gr.Tab("Traffic Flow Map"):
                    with gr.Group():
                        gr.Label(value="Traffic Flow Conditions",show_label=False)
                        trafficFlowDropdown=gr.Dropdown(["__Most Current__"]+trafficFlowDAO.getDateList(),value="01/01/2024 7:00",label="Select filter to get data by date&time")
                        map=gr.Plot(show_label=False)
                demo.load(filter_map,outputs=map)
                trafficFlowDropdown.change(filter_map,inputs=trafficFlowDropdown,outputs=map)

                with gr.Tab("Crisis Response",id="QuickQuery"):
                    with gr.Group():
                        with gr.Row():
                            with gr.Row():
                                quickQueryFromTo=gr.Dropdown(choices=["FROM","TO"],value="FROM",interactive=False,show_label=False,scale=1,min_width=60,filterable=False)
                                quickQueryDorpDown=gr.Dropdown(choices=[
                                    "hospital",
                                    "fire station",
                                    "police station",
                                    "bus interchange"
                                ],value="hospital",interactive=True,show_label=False,scale=2,min_width=100,filterable=False)
                            with gr.Row():
                                quickQueryToFrom=gr.Dropdown(choices=["TO","FROM"],value="TO",interactive=True,show_label=False,scale=1,min_width=60,filterable=False)
                                quickQueryTextbox=gr.Textbox(show_label=False,scale=2,min_width=100,placeholder="Type here&Enter")
                        quickQueryHtmlBox=gr.HTML(generateHTMLText(None))
                with gr.Tab("Open Query",id="FreeQuery") as freeQueryTab:
                    with gr.Row():
                        with gr.Column(min_width=120):
                            freeQueryReportArea=gr.Chatbot(
                                value=[
                                    [
                                        None,
                                        """
                                        Hello, I'm an emergency event report chatbot. You can either:\n
                                        1) Describe a public traffic emergency, including what is happening and at what location, or\n
                                        2) Query for a route from one location to another\n
                                        \n
                                        The traffic condition from SMU to NUS is shown in the map.
                                        """
                                    ]
                                ],
                                scale=1,
                                min_width=120,
                                show_label=False
                            )
                            freeQueryTextbox=gr.Textbox(show_label=False,placeholder="Enter to submit.Please ask for a route",scale=1,min_width=120)
                        
                        with gr.Column(min_width=1080):
                            with gr.Group():
                                jsonTextBufferForReportReaderReturnList=gr.Textbox(value=None,visible=False)
                                freeQueryMapDropdown=gr.Dropdown(scale=3,show_label=False,interactive=False)
                                freeQueryHtmlBox=gr.HTML(generateHTMLText(None))


                    quickQueryToFrom.change(changeQuickQueryToFrom,quickQueryToFrom,quickQueryFromTo)
                    quickQueryTextbox.submit(submitQuickQueryTextbox,[quickQueryFromTo,quickQueryDorpDown,quickQueryToFrom,quickQueryTextbox],quickQueryHtmlBox)

                    freeQueryTextbox.submit(submitFreeQueryTextbox,inputs=[freeQueryReportArea,freeQueryTextbox],outputs=[freeQueryReportArea,freeQueryHtmlBox,freeQueryMapDropdown,jsonTextBufferForReportReaderReturnList])
                    freeQueryMapDropdown.input(changeFreeQueryDropDown,inputs=[freeQueryMapDropdown,jsonTextBufferForReportReaderReturnList],outputs=freeQueryHtmlBox)
                    
                with gr.Tab("Sustainability Monitoring"):
                    with gr.Group():
                        final_year = gr.Radio([2025, 2026, 2027, 2028, 2029, 2030], label="Forecast to:")
                        show_legend = gr.Checkbox(label="Show Legend")
                        submit = gr.Button("Submit")
                        output_plot = gr.Plot(show_label=False)
                        submit.click(sustainability_forecast.plot_dataframes, [final_year, show_legend], output_plot)


demo.launch()