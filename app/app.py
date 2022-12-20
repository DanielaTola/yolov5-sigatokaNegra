import argparse
import io
import os
from PIL import Image
import torch
from flask import Flask, render_template, request, redirect
import base64
import tensorflow as tf 

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
           return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])
        
        results.render()# updates results.imgs with boxes and labels
        print(results)
        results.save("static/image0.jpg")


        #copia codigo prueba bse 64
        base64_image = results
        decode_base64 = base64.b64decode(base64_image)
        print(decode_base64)
        return render_template("index.html", results=decode_base64)
    else:
        return render_template("index.html")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load(
       "ultralytics/yolov5", "custom", path= 'C://Users//User/Desktop//tesis Daniela//trainingModel//best.pt', force_reload=True)# force_reload = recache latest code
    model.eval()
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    #model.eval()
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat