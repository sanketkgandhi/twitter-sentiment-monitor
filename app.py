import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import pandas as pd
import plotly.graph_objs as go
import math
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import tweepy
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',"https://unpkg.com/tachyons@4.10.0/css/tachyons.min.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.scripts.config.serve_locally = True
app.title = 'Twitter Sentiment Monitor'

consumer_key = 'qhwg88DbtCpCG2hQumqSKj3qp'
consumer_secret = 'BZy237443Jj7hePJvSnRUFMePKCnrXVtEbhzXYQbDEwtUm8LQy'
access_token = '3315982002-w8V3IgrJWXKjNHuKjqMWOj7dGsTqG2rZaQSl91Y'
access_token_secret = 'LBkNkwRhZga9O3MokOIClPagFWGBx97DDo6RXvWFwqjrv'


app.layout =html.Div(children=[
    html.H1('Twitter Sentiment Analysis ', 
        style={
            'textAlign': 'center',
            'font-size': '4.5rem',
            'font-weight': '300',
            'line-height': '1.2',
            'font':'aileron'
    }),
    html.Div(children=[
        html.Div(children=[
            html.Span(children="#",
                style={'display': 'flex',
                'align-items': 'center',
                'padding':' .375rem .75rem',
                'line-height': '1.5;',
                'color':' #495057;',
                'text-align': 'center',
                'background-color':' #e9ecef',
                'box-shadow':' 3px 3px 8px 0 #c1c1c1',
                
                })],
            style={'margin-right': '-1px',
                    'display': 'flex'}),
        dcc.Input(id="my_inp",value="",placeholder="Enter HashTag to Analyse",type="text",
            style={ 'position': 'relative',
                'flex':' 1 1 auto;',
                'border-radius': '5px',
                'box-shadow': '3px 3px 8px 0 #c1c1c1',
                'width':'100%',
                'margin-bottom': '0',
                'display': 'block',
                'padding':' .375rem .75rem',
                'font-size':' 1.7rem',
                'font-weight': '400',
                'line-height': '1.5',
                'color': '#495057',
                'background-color': '#fff',
                'background-clip': 'padding-box',
                }),
        html.Button(id="sub-btn" ,children="Analyze", n_clicks=0,style={'background-color':'rgb(153, 255, 153)','margin-left':'5%'})
        ],

        style={ 'margin-bottom': '1rem !important',
                'position':'absolute',
                'display': 'flex',
                'left':'50%',
                'transform': 'translate(-50%)',
                'width':'40%'
                
                }),
    html.Br(),html.Br(),

    html.Div(children=[dcc.Loading(id='loading',children=[html.Div([html.Div(id="live-update-graph")])],
                                        type='graph',fullscreen=False,
                                        style={"position":"fixed",'top':'50%','left':'50%',
                                                'width': '100%','transform':'translate(-50%, -50%)' })
                        ],style={ 'background-color': '#f1f1f1'}),

    html.Footer(children="Developed By Sanket",style={ 'bottom':'0px','position':'fixed','margin-bottom': 'auto !important',
                                                            'background-color': '#f9f9f9',
                                                            'height': '30px',
                                                            'width': '100%',
                                                            'padding':' 8px 0 0',
                                                            'text-align':'center'
                                                            })
    
    ],style={})

    



@app.callback(Output('live-update-graph', 'children'),
                [Input(component_id="sub-btn",component_property="n_clicks" )],
                [State('my_inp','value')]
                )
def update_graph_live(n_clicks,n):
    if (len(n)!=0):
        global graph
        graph = tf.compat.v1.get_default_graph()
        model = load_model('Sentiment_LSTM_model.h5')
        MAX_SEQUENCE_LENGTH = 300

        # Twitter
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth,wait_on_rate_limit=True)
        with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)


        def predict(text, include_neutral=True):
            # Tokenize text
            x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_SEQUENCE_LENGTH)
            # Predict
            score = model.predict([x_test])[0]
            if(score >=0.4 and score<=0.6):
                label = "Neutral"
                polarity=0
            if(score <=0.4):
                label = "Negative"
                polarity=-1
            if(score >=0.6):
                label = "Positive"
                polarity=1
            return {"label" : label,
                "score": float(score),"polarity":polarity}

        tweets = []
        for tweet in tweepy.Cursor(api.search,q="#" + n + " -filter:retweets",rpp=5,lang="en", tweet_mode='extended').items(50):
            temp = {}
            #print(len(n),"zest")
            temp["created_at"] = tweet.created_at
            temp["text"] = tweet.full_text
            temp["username"] = tweet.user.screen_name
            temp["user_location"]=tweet.user.location
            temp["id_str"]=tweet.user.id_str
            with graph.as_default():
                prediction = predict(tweet.full_text)
            temp["label"] = prediction["label"]
            temp["score"] = prediction["score"]
            temp["polarity"] = prediction["polarity"]
            
            tweets.append(temp)
        #print(n,"abcdefghijkl")
        TRACK_WORD=n
        df=pd.DataFrame(tweets)
        df['created_at'] = pd.to_datetime(df['created_at'])
        result = df.groupby([pd.Grouper(key='created_at', freq='60s'), 'polarity']).count().unstack(fill_value=0).stack().reset_index()
        result = result.rename(columns={ "id_str": "Num of '{}' mentions".format(TRACK_WORD),"created_at":"Time in UTC" })
        time_series = result["Time in UTC"][result['polarity']==0] .reset_index(drop=True)
        
        neu_num = (df['polarity']==0).sum()
        neg_num = (df['polarity']==-1).sum()
        pos_num = (df['polarity']==1).sum()
        
        ################################################
        content =' '.join([t['text'] for t in tweets])
        content = re.sub(r"http\S+", "", content)
        content = content.replace('RT ', ' ').replace('&amp;', 'and')
        content = re.sub('[^A-Za-z0-9]+', ' ', content)
        content = content.lower()

        tokenized_word = word_tokenize(content)
        stop_words=set(stopwords.words("english"))
        filtered_sent=[]
        for w in tokenized_word:
            if (w not in stop_words) and (len(w) >= 3):
                filtered_sent.append(w)
        fdist = FreqDist(filtered_sent)
        fd = pd.DataFrame(fdist.most_common(10),columns = ["Word","Frequency"]).drop([0]).reindex()
        fd['Polarity'] = fd['Word'].apply(lambda x: TextBlob(x).sentiment.polarity)
        fd['Marker_Color'] = fd['Polarity'].apply(lambda x: 'rgba(192,0,0, 0.6)' if x < -0.1 else \
            ('rgba(153, 255, 153, 0.6)' if x > 0.1 else 'rgba(77, 77, 255, 0.6)'))
        fd['Line_Color'] = fd['Polarity'].apply(lambda x: 'rgba(192,0,0, 1)' if x < -0.1 else \
            ('rgba(153, 255, 153, 1)' if x > 0.1 else 'rgba(77, 77, 255, 1)'))

        countries = ['United States of America','USA', 'United Kingdom','UK', 'Jamaica',
           'Czech Republic', 'New Zealand', 'China', 'Canada', 'Germany',
           'Japan', 'France', 'Australia', 'Italy', 'Spain', 'India',
           'Belgium', 'nan', 'Hong Kong', 'Norway', 'Ireland', 'South Africa',
           'Mexico', 'Malaysia', 'Finland', 'Iceland', 'Denmark',
           'Philippines', 'Russia', 'Bulgaria', 'Switzerland',
           'United Arab Emirates', 'Malta', 'South Korea', 'Brazil', 'Peru',
           'Netherlands', 'Bosnia and Herzegovina', 'Luxembourg', 'Romania',
           'Singapore', 'Aruba', 'Libyan Arab Jamahiriya', 'Hungary',
           'Argentina', 'Panama', 'Austria', 'Greece', 'Sweden', 'Thailand',
           'Fiji', 'Bahamas', 'Turkey', 'Cyprus', 'Bolivia', 'Morocco',
           'Ecuador', 'Poland', 'Israel', 'Bhutan', 'Lebanon',
           'Kyrgyz Republic', 'Algeria', 'Indonesia', 'Guyana', 'Pakistan',
           'Guadaloupe', 'Iran', 'Slovenia', 'Afghanistan',
           'Dominican Republic', 'Cameroon', 'Kenya']
        india_states=['Andhra Pradesh','AP','Arunachal Pradesh','AR','Assam','AS','Bihar','BR','Chhattisgarh','CG','Goa','GA','Gujarat','GJ','Haryana','HR','Himachal Pradesh','HP','Jammu and Kashmir','JK','Jharkhand','JH',
        'Karnataka','KA','Kerala','KL','Madhya Pradesh','MP','Maharashtra','MH','Manipur','MN','Meghalaya','ML','Mizoram','MZ','Nagaland','NL','Orissa','OR','Punjab','PB','Rajasthan','RJ','Sikkim','SK','Tamil Nadu','TN','Tripura','TR','Uttarakhand','UK','Uttar Pradesh','UP','WestBengal','WB','Telangana','TS','Andaman and Nicobar','AN',
        'Chandigarh','CH','Dadra and Nagar Haveli','DH','Daman and Diu','DD','Delhi','DL','Lakshadweep','LD','Pondicherry','PY'
        ]
        usa_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
                  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
                  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
                  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
                  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY","Alabama","Alaska","Arizona","Arkansas","California","Colorado",
          "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
          "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
          "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
          "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
          "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
          "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
          "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]
        loc=[ltag['user_location'] for ltag  in tweets]
        #print(loc)
        is_in_list=[]
        #geo = a['user_location']
        #a = a.fillna(" ")
        for x in loc:
            check = False
            for s in countries:
                if s in x:
                    is_in_list.append(s)
                    check = True
                    break
            if not check:
                for q in india_states:
                    if q in x:
                        is_in_list.append("India")
                        check=True
            if not check:
                for w in usa_states:
                    if w in x:
                        is_in_list.append("USA")
                        check=True
               # is_in_list.append(None)
        
        geo_dist = pd.DataFrame(is_in_list, columns=['State']).dropna().reset_index()
        geo_dist = geo_dist.groupby('State').count().rename(columns={"index": "Number"}).sort_values(by=['Number'], ascending=False).reset_index()
        geo_dist["Log Num"] = geo_dist["Number"].apply(lambda x: math.log(x, 2))
        geo_dist['Full State Name'] = geo_dist['State']#.apply(lambda x: INV_STATE_DICT[x])
        geo_dist['text'] = geo_dist['Full State Name'] + '<br>' + 'Num: ' + geo_dist['Number'].astype(str)
        
        ################################################
        # Create the graph 
        children =[
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='crossfilter-indicator-scatter',
                                figure={
                                    'data': [
                                        go.Scatter(
                                            x=time_series,
                                            y=result["Num of '{}' mentions".format(TRACK_WORD)][result['polarity']==0].reset_index(drop=True),
                                            name="Neutrals",
                                            opacity=1,
                                            mode='lines',
                                            line=dict(width=0.5, color='rgb(77, 77, 255)'),
                                            stackgroup='one' 
                                        ),
                                        go.Scatter(
                                            x=time_series,
                                            y=result["Num of '{}' mentions".format(TRACK_WORD)][result['polarity']==-1].reset_index(drop=True).apply(lambda x: -x),
                                            name="Negatives",
                                            opacity=0.8,
                                            mode='lines',
                                            line=dict(width=0.5, color='rgb(192,0,0)'),
                                            stackgroup='two' 
                                        ),
                                        go.Scatter(
                                            x=time_series,
                                            y=result["Num of '{}' mentions".format(TRACK_WORD)][result['polarity']==1].reset_index(drop=True),
                                            name="Positives",
                                            opacity=0.6,
                                            mode='lines',
                                            line=dict(width=0.5, color='rgb(153, 255, 153)'),
                                            stackgroup='three' 
                                        )
                                    ],'layout':go.Layout(title='Tweets Scatterplot',
                                        xaxis={'title':'Time in UTC'},yaxis={"title":"Number of {} mentions".format(TRACK_WORD)},
                                        plot_bgcolor='#f1f1f1',
                                        paper_bgcolor='#f1f1f1')
                                }
                            )
                        ], style={'width': '73%', 'display': 'inline-block', 'padding': '0 0 0 20','background-color': '#f1f1f1'}),
                        
                        html.Div([
                            dcc.Graph(
                                id='pie-chart',
                                figure={
                                    'data': [
                                        go.Pie(
                                            labels=['Positives', 'Negatives', 'Neutrals'], 
                                            values=[pos_num, neg_num, neu_num],
                                            name="View Metrics",
                                            marker_colors=['rgba(153, 255, 153, 0.6)','rgba(192,0,0, 0.6)','rgba(77, 77, 255, 0.6)'],
                                            #textinfo='value',
                                            hole=.65)
                                    ],
                                    'layout':{
                                        'showlegend':False,
                                        'title':'',
                                        'annotations':[
                                            dict(
                                                text='{0:.1f}K'.format((pos_num+neg_num+neu_num)/1000),
                                                font=dict(
                                                    size=40
                                                ),
                                                showarrow=False
                                            )
                                        ],
                                        'plot_bgcolor':'#f1f1f1',
                                        'paper_bgcolor':'#f1f1f1'
                                    }

                                }
                            )
                        ], style={'width': '27%', 'display': 'inline-block'})
                    ]),
                    html.Div([
                            dcc.Graph(
                                id='x-time-series',
                                figure = {
                                    'data':[
                                        go.Bar(                          
                                            x=fd["Frequency"].loc[::-1],
                                            y=fd["Word"].loc[::-1], 
                                            name="Neutrals", 
                                            orientation='h',
                                            marker_color=fd['Marker_Color'].loc[::-1].to_list(),
                                            marker=dict(
                                                line=dict(
                                                    color=fd['Line_Color'].loc[::-1].to_list(),
                                                    width=1),
                                                ),
                                        )
                                    ],
                                    'layout':go.Layout(title='',
                                            xaxis={'title':'Frequency'},
                                            plot_bgcolor='#f1f1f1',
                                            paper_bgcolor='#f1f1f1')
                                }        
                            )
                        ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 0 0 20'}),

                        html.Div([
                            dcc.Graph(
                                id='y-time-series',
                                figure = {
                                    'data':[
                                        go.Choropleth(
                                            locations=geo_dist['State'], # Spatial coordinates
                                            z = geo_dist['Log Num'].astype(float), # Data to be color-coded
                                            locationmode = 'country names', # set of locations match entries in `locations`
                                            #colorscale = "Blues",
                                            text=geo_dist['text'], # hover text
                                            geo = 'geo',
                                            colorbar_title = "Num in Log2",
                                            marker_line_color='white',
                                            colorscale = ["#5bd75b", "#0f3e0f"],
                                            #autocolorscale=False,
                                            #reversescale=True,
                                        ) 
                                    ],
                                    'layout': {
                                        'title': "Geographic Distribution ",
                                        'geo':{'scope':'world'},
                                        'plot_bgcolor':'#f1f1f1',
                                        'paper_bgcolor':'#f1f1f1'
                                    }
                                }
                            )
                        ], style={'display': 'inline-block', 'width': '49%'})

                    
                    
                ]
        return children




if __name__ == '__main__':
    app.run_server()
