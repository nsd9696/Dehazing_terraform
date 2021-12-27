import os
# get_ipython().system_raw('/Users/sangdae/Documents/GitHub/konkuk_graduate_project/ngrok http 8501 &')
# os.system("ngrok authtoken 1zWWMbQ02vhR7L7jHNCiOC5R1w4_4Rbm8NotooxBD7Aaxq8X7")
os.system('./ngrok http 8501 &')
os.system('''
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    'import sys, json; print("Execute the next cell and the go to the following URL: " +json.load(sys.stdin)["tunnels"][0]["public_url"])'
''')
os.system('''streamlit run app.py''')

# from pyngrok import ngrok 
# import subprocess
# import os
# os.system("ngrok authtoken 1zWWMbQ02vhR7L7jHNCiOC5R1w4_4Rbm8NotooxBD7Aaxq8X7")
# os.system("nohup streamlit run app.py")

# url = ngrok.connect(port = 8501)
# print(url) #generates our URL

# os.system("streamlit run --server.port 80 app.py >/dev/null") #used for starting our server