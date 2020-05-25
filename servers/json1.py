import requests

url='http://127.0.0.1:5000/predict-api'
r=requests.post(url,json={'size':'2','cylinder':'6','fuel':'9.6'})
print(r.json())
