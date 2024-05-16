FROM debian:latest

# Update the package list and install necessary dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-venv python3-pip pkg-config default-libmysqlclient-dev build-essential iputils-ping curl
    
# Create a working directory
WORKDIR /usr/src/app

# Copy the local files to the container
COPY . /usr/src/app

# Set the timezone to US Central Time (UTC-6)
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create a virtual environment
RUN python3 -m venv venv

# Activate the virtual environment
ENV PATH="/usr/src/app/venv/bin:$PATH"

# Install the required Python packages in the virtual environment
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the default command to run the program
CMD ["python3", "long_term_test.py"]