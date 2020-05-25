# Analysis of Regression Algorithms to Predict Future Weather

### Summary
This repository contains a report of three regression algorithms used to predict weather based on historical Siberian weather data: Kernel Ridge Regression, k-Neighbors Regression, and Neural Networks. Also included in the repository is the code used to generate data for the report.

### Instructions on dataset
The dataset I used in my experiments (weather.csv) is attached to this zip file. It can be accessed at https://www.kaggle.com/dwdkills/weather, but I ran some processing on the original dataset because it originally used commas as decimal points (European standard); therefore, I changed all the commas to periods and included the updated dataset in this zip file.

### Instructions on how to run
In its current state, the main.py program can test all three regression algorithms and generate all graphs and figures without commenting or uncommenting any code. If any specific algorithm is desired, simply uncomment the other two algorithms.

The main packages I used for my program are: pandas, numpy, matplotlib (pyplot), and sklearn. To install the packages, use: 'pip3 install ['package name']'.
