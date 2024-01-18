import requests
url = 'https://predict-nzzyxeyodq-ew.a.run.app/predict_image/'

# %%
for _ in range(100):

    # Open the image file in binary mode
    image = open("test.png", "rb")
    # Define the files dictionary
    files = {"file": ("test.png", image, "image/png")}
    r = requests.post(url, files=files)
    image.close()
    assert r.json() == {'class_id': [3]}
