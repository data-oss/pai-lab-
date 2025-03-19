from flask import Flask, render_template, request
import requests

app = Flask(__name__)

NASA_API_KEY = "Ww9qA5ZauqnfRhHZMdj3nfm3pArw9uIQQm0ocnCb"
NASA_URL = "https://api.nasa.gov/planetary/earth/imagery"

def get_nasa_image(lat, lon, date, dim=0.1):
    params = {
        "lat": lat,
        "lon": lon,
        "date": date,
        "dim": dim,
        "api_key": NASA_API_KEY
    }
    response = requests.get(NASA_URL, params=params)
    if response.status_code == 200:
        return response.url
    return None

@app.route("/", methods=["GET", "POST"])

def index():
    image_url = None
    if request.method == "POST":
        lat = request.form.get("lat")
        lon = request.form.get("lon")
        date = request.form.get("date")
        image_url = get_nasa_image(lat, lon, date)
    return render_template("index.html", image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
