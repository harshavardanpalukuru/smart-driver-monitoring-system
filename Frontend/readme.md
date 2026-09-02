# Smart Driver Monitoring System

An AI-powered driver monitoring system that detects unsafe driving behaviors in real time using YOLO and Flask.

## Features

- Real-time driver behavior detection
- Detection of distracted driving
- Drowsiness detection
- Phone usage detection
- Drinking detection
- Eating detection
- Smoking detection
- Seatbelt detection
- Email alerts for unsafe behavior
- Audio alerts
- User registration and login
- MySQL database integration

## Technologies Used

- Python
- Flask
- YOLO
- OpenCV
- MySQL
- HTML
- CSS
- JavaScript
- Pygame

## Setup


### 1. Create a virtual environment

```bash
python -m venv venv
````

### 2. Activate the virtual environment

For Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```cmd
venv\Scripts\activate
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the environment variables

Create a `.env` file in the project root directory and add the required database and email configuration.

Example:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_PORT=3306
DB_NAME=smartdriver

EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_email_app_password
```

Do not share or commit the `.env` file.

### 5. Set up the MySQL database

Open MySQL and create the database and `users` table using the SQL commands in:

```text
db.sql
```

The application uses the `smartdriver` database.

### 6. Run the Flask application

Make sure the virtual environment is activated, then run:

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

### 7. Open the application

Open the following address in your web browser:

```text
http://127.0.0.1:5000
```

Register a user account, log in, and start the driver monitoring system.

```


