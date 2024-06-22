
# HousePriceEstimationProject

## Overview
The HousePriceEstimationProject aims to estimate house prices using various machine learning techniques. This project involves data collection, cleaning, model training, and deployment through a FastAPI service.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- Data crawling and collection
- Data cleaning and preprocessing
- Various machine learning models for price estimation
- API service for predictions using FastAPI

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/javad787/HousePriceEstimationProject.git
    cd HousePriceEstimationProject
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare the data:
    ```bash
    python data_cleaning/clean_data.py
    ```
2. Train the models:
    ```bash
    python models/train_model.py
    ```
3. Run the FastAPI server:
    ```bash
    uvicorn fastapi.main:app --reload
    ```
4. Access the API documentation at `http://127.0.0.1:8000/docs`

## Project Structure
```
HousePriceEstimationProject/
│
├── data/                # Raw and processed data
├── data_cleaning/       # Scripts for cleaning data
├── data_crawling/       # Scripts for collecting data
├── fastapi/             # FastAPI application
├── models/              # Machine learning models
├── .gitignore           # Git ignore file
├── README.md            # Project README
├── requirements.txt     # Python dependencies
```

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
