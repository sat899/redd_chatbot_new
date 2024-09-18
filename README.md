# redd_chatbot

## running the app locally

First extract the files from the final_project folder and put them all in a single folder of your choosing. Details of all the files needed are provided in Catalog.pdf.

Next, ensure that you are running the correct version of Python, which is 3.9.7.

Then, in your terminal create a virtual environment and install the required packages:

```
python -m venv redd_chatbot_venv
source redd_chatbot_venv/bin/activate
pip install -r requirements.txt
```

Next, run the app using the command:

```
streamlit run app.py
```

## running the testing.ipynb script

The best way to test the model is to use the testing.ipynb notebook. If you have successfully run the app, you should also be able to run this notebook too.

Simply open this file in an IDE such as VSCode and press 'Run All'. The cells will all run and print out various outputs showing the contents of the various variables and prompt strings. At the bottom of the script you will be able to see the full prompt that gets provided to the model and the response it generates.

The solution can be tested for different users by modifying the email address in the sixth cell. A full list of the email addresses available in the testing dataset can be found in testing.xlsx.

## manually testing through the interface

An alternative way to test the tool is directly in the interface through prompting.

Some examples of prompts and their expected outputs have been provided in the file 'test_examples.docx'. 

These prompts can be directly copied into the user input field in the application. You can then press 'send' to see the output the assistant provides and compare this to the expected one in the document.

Further test prompts can be developed quite easily by reviewing the user information in 'testing.xlsx' and adapting the prompts to fit these.