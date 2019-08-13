import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.svm import SVR
from scipy.optimize import fsolve


cities = [
    'Mumbai',
    'Thane',
    'Badlapur',
    'Pune',
    'Solapur',
    'Shirdi',
    'Ahmednagar',
    'Ratnagiri',
    'Aurangabad',
    'Ichalkaranji'
]

distances = {
    'Mumbai': 0,
    'Thane': 27,
    'Badlapur': 49,
    'Pune': 120,
    'Solapur': 182,
    'Shirdi': 195,
    'Ahmednagar': 202,
    'Ratnagiri': 223,
    'Aurangabad': 261,
    'Ichalkaranji': 304
}

dates = [x for x in range(1, 21)]


def get_data(city, date1='2019-07-01', date2='2019-07-20', api_key = ''):
    api_link = ('http://api.worldweatheronline.com/premium/v1/past-weather' +
                '.ashx?key={}&q={}' +
                '&format=json&date={}&enddate={}')
    try:
        values = requests.get(api_link.format(api_key, city, date1, date2))
    except:
        print("Invalid City")
        return 0

    values = values.text
    values = json.loads(values)
    '''
    What we get from the website, in the values
    variable is a dict, which has one key called data,
    the value of this key data is another dict,
    This dict contains 2 keys :
    weather and request
    request contains information about the call
    to the api
    the weather contains the actual values of weather
    you had requested.
    The value of the weather key is a list.
    Continued later
    '''
    return values['data']['weather']


def clean_data(values):
    '''
    Info:
        -- Continued --
        values is a list of dicts
        each dict is another day
        each day has a key called hourly
        (among other keys, like avgtemp)
        whose value is a dict of 8 readings
        of temperatures throughout the day.
        These 8 reading are in the form of
        dicts with readings in order,
        with 3hr differences
        so 24hrs / 8 reading = 3 hour gaps.
        '''
    data = [{
        'avgtemp': '',
        'temperature': [],
        'humidity': [],
        'pressure': [],
        'weather_description': [],
        'wind_speed': [],
        'wind_degree': [],
        'hour': [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400]
    } for i in range(20)]

    # Created a list of 20 dicts for 20 days

    features = {
        'temperature': 'tempC',
        'humidity': 'humidity',
        'pressure': 'pressure',
        'weather_description': 'weatherDesc',
        'wind_speed': 'windspeedKmph',
        'wind_degree': 'winddirDegree'
    }

    for each_day, current_data in zip(data, values):
        for j in range(8):
            each_day['avgtemp'] = (current_data['avgtempC'])
            for item in features:
                if item == 'weather_description':
                    each_day[item].append(
                        current_data['hourly'][j][features[item]][0]['value'])
                else:
                    each_day[item].append(float(
                        current_data['hourly'][j][features[item]]))

    return data


def save_cities_data():
    for city in cities:
        city_data = get_data(city)
        with open(city+"Report.txt", 'w') as f:
            f.write(str(city_data))


def load_city_data(city):
    with open(city+"Report.txt") as f:
        data = eval(f.read())

    return data


def final_data():
    data = {}
    for city in cities:
        print("Working on ", city)
        temp = load_city_data(city)
        print("Loaded Data")
        data[city] = clean_data(temp)
        print("Completed", city)
        print()
    return data


def plot_graph_1(plot_type, top_label, y_label):

    close_and_far = ['Mumbai',
                     'Thane',
                     'Badlapur',
                     'Ratnagiri',
                     'Aurangabad',
                     'Ichalkaranji']
    data = final_data()
    x_labels = [str(x)+":00" for x in range(0, 24, 3)]
    y_plots = []
    # {[{[]},],}
    for city in close_and_far:
        day = data[city][0]
        y_plots.append(day[plot_type])
    count = 1
    x_vals = [x for x in range(8)]
    for plots in y_plots:
        if count < 4:
            color = 'g'
            label = 'Close to sea'
        else:
            color = 'r'
            label = "Far from sea"
        print(count, color, plots)
        if count in (3, 4):
            plt.plot(x_vals, plots, color=color, label=label)
        else:
            plt.plot(x_vals, plots, color=color)
        count += 1
    plt.title(top_label)
    plt.xlabel("Time")
    plt.ylabel(y_label)
    plt.xticks(np.arange(8), x_labels, rotation=290)
    plt.grid(True)
    plt.legend()
    plt.show()


def max_and_mins(plot_type):
    data = final_data()

    maximun = []
    minimum = []
    dist = []

    for city in data:
        day = data[city][0]
        temp = day[plot_type]
        maximun.append(max(temp))
        minimum.append(min(temp))

    for city in cities:
        dist.append(distances[city])

    return maximun, minimum, dist


def plot_graph_2(max_min, plot_type, title, y_label):

    maximun, minimum, dist = max_and_mins(plot_type)

    # plt.plot(dist, maximun, 'ro', color='r', label="Max temps")
    # plt.plot(dist, minimum, 'ro', color='g', label="Min temps")

    plt.title(title)

    plt.xlabel("Distance in Km")
    plt.ylabel(y_label)

    x = np.array(dist)
    if max_min.lower() == "max":
        y = np.array(maximun)
    if max_min.lower() == "min":
        y = np.array(minimum)

    x1 = x[x < 100]
    x1 = x1.reshape((x1.size, 1))

    y1 = y[x < 100]

    x2 = x[x > 50]
    x2 = x2.reshape((x2.size, 1))

    y2 = y[x > 50]

    svr_line1 = SVR(kernel="linear", C=1e3)
    svr_line2 = SVR(kernel="linear", C=1e3)

    svr_line1.fit(x1, y1)
    svr_line2.fit(x2, y2)

    x_predict1 = np.arange(10, 100, 10).reshape((9, 1))
    x_predict2 = np.arange(50, 400, 50).reshape((7, 1))

    y_predict1 = svr_line1.predict(x_predict1)
    y_predict2 = svr_line2.predict(x_predict2)

    plt.plot(x_predict1, y_predict1, c='r', label='Strong sea effect')
    plt.plot(x_predict2, y_predict2, c='b', label='Light sea effect')
    plt.axis((0, 400, 27, 32))
    plt.scatter(x, y, c='purple', label='data')

    print(svr_line1.coef_)
    print(svr_line1.intercept_)
    print(svr_line2.coef_)
    print(svr_line2.intercept_)

    def line1(x):
        m = svr_line1.coef_[0][0]
        c = svr_line1.intercept_[0]
        return m*x + c

    def line2(x):
        m = svr_line2.coef_[0][0]
        c = svr_line2.intercept_[0]
        return m*x + c

    def POI(fun1, fun2, x0):
        return fsolve(lambda x: fun1(x) - fun2(x), x0)

    result = POI(line1, line2, 0.0)
    print("[x,y] = [ %d , %d ]" % (result, line1(result)))

    x = np.linspace(0, 300, 31)
    plt.plot(x, line1(x), x, line2(x), result, line1(result), 'ro')

    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    plot_graph_1('temperature',
                 'Temperature of Six Cities\n3 far from sea and 3 close',
                 'Temp in C')
    plot_graph_2('max', 'temperature',
                 " Max Temperature of 10 cities",
                 "Temp in C")
    plot_graph_2('min', 'temperature',
                 " Min Temperature of 10 cities",
                 "Temp in C")
    plot_graph_1('humidity',
                 'Humidity of Six Cities\n3 far from sea and 3 close',
                 'Humidity')
    plot_graph_2('max', 'humidity',
                 " Max Humidity of 10 cities",
                 "Humidity")
    plot_graph_2('min', 'humidity',
                 " Min Humidity of 10 cities",
                 "Humidity")


if __name__ == '__main__':
    main()
