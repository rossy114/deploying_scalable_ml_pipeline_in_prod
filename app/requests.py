import requests

response = requests.post('https://ml-dev-ops-salary-app.herokuapp.com/')

print(response.status_code)
print(response.json())