from flask import Flask, render_template, request
from flask import Flask

from utils.functions import *

app = Flask(__name__)



@app.route("/", methods=['POST', 'GET'])
def index():



    return render_template("index.html")

@app.route("/visualize", methods=['POST', 'GET'])
def visualize():
    if request.method == 'POST':
        export_url = request.form['export_url']
        if 'https://lichess.org/api/games/user/' not in export_url:
            return "provide proper url"
        player = export_url.split('/')[6].split('?')[0]
        df = build_df(export_url)
        print('df built')
        df = build_data(df, player)
        print('data built')
        basic_headings, data = get_basic_data(df)
        game_hour = game_by_hour(df)

        print('basics built')
        if 'Opening' in df.columns:
            opening_headins, opening_data, opening_plot = get_openings(df)
            print('openings built')
        rematch_data = rematch(df, player)
        print('rematch built')
        if len(rematch_data) > 0:
            rem = 1
        else:
            rem = 0   

        mean = monthly_mean(df)        
        img = plot_lr(df)
        print('lr plot built')


    return render_template("data.html", 
                            headings=basic_headings, 
                            data=data,
                            op_headings=opening_headins,
                            op_data=opening_data,
                            re=rematch_data,
                            rem=rem,
                            img = img,
                            monthly_mean = mean,
                            opening_vis = opening_plot,
                            hour = game_hour
                            )


if __name__ == '__main__':
    app.run()
