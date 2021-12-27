# lightweight python
FROM ubuntu:20.04

RUN apt-get update && apt-get install python3 python3-pip -y

EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run app.py --server.enableCORS=false


# # Copy local code to the container image.
# # ENV APP_HOME /app
# EXPOSE 8501
# WORKDIR /app
# COPY . ./

# # RUN ls -la $APP_HOME/

# # Install dependencies
# RUN pip install -r requirements.txt

# # Run the streamlit on container startup
# CMD [ "streamlit run app.py" ]