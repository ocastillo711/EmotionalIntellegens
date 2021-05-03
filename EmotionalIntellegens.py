import PySimpleGUI as sg
from deepface import DeepFace
from onboarding import subject_onboarding

"""
Omar Castillo
Final Project Submitted in Partial Fulfilment of the Requirement for the Award of the Degree of Master of Science, Software Engineering.
CPSC 597
Spring 2021
"""

"""
This is the home screen for the Emotional Intellegens Application. This home screen launches the various modules
contained in the project. The User can run the application by completing three simple steps needed to run the application.
"""

# Initialize Graphics
splash_screen_logo = r'C:\Users\odb88\PycharmProjects\EmotionalIntellegens\Graphics\SplashScreen Logo.png'
icon_logo64 = b'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'
icon_logo = r'C:\Users\odb88\PycharmProjects\EmotionalIntellegens\Graphics\IconLogo.png'
home_screen_logo = r'C:\Users\odb88\PycharmProjects\EmotionalIntellegens\Graphics\home_screen_logo.png'

# Set Window Theme
sg.theme('Reddit')

# Splash Screen
DISPLAY_TIME_MILLISECONDS = 1000
sg.Window('Window Title', [[sg.Image(splash_screen_logo)]], no_titlebar=True,
          keep_on_top=True).read(timeout=DISPLAY_TIME_MILLISECONDS, close=True)
# Set Layout
layout = [
    # Emotional Intellegens Logo
    [sg.Image(home_screen_logo)],
    # Step 1
    [sg.Frame(layout=[
        [sg.Text('Subject Name:'), sg.InputText(size=(20, 1), key='-INPUT-'), sg.Submit()],
        [sg.Text('Subject Name:', pad=(5, 10)), sg.Text(size=(20, 1), key='-OUTPUT-')]], title='Step 1:'
    )],

    # Step 2
    [sg.Frame(layout=[
        [sg.Button('Onboard Subject', pad=(150, 5))],
        [sg.Button("Upload Subject's Image", pad=(120, 15))]], title='Step 2:'
    )],

    # Step 3
    [sg.Frame(layout=[
        [sg.Button('Start Human Emotion Detection', pad=(87, 15))]], title='Step 3:'
    )],

    [sg.Button('Select Camera', pad=(0, 10))],

    [sg.Exit(pad=(0, 10))]
]

# Create Windows
window = sg.Window('Emotional Intellegens',
                   default_element_size=(21, 1),
                   auto_size_text=True, font='helvetica', element_justification='center', icon=icon_logo64).Layout(layout)

# Event Loop
while True:
    event, values = window.Read()
    if event in (None, 'Exit'):
        break

    if event == 'Submit':
        # Update the "output" text element to be the value of "input" element
        window['-OUTPUT-'].update(values['-INPUT-'])

    elif event == 'Start Human Emotion Detection':
        DeepFace.stream("DATABASE")
        text_input = values['-INPUT-']

    elif event == "Upload Subject's Image":
        sg.popup_get_file('Select Image:', "Upload Subject's Image", icon=icon_logo64)

    elif event == 'Onboard Subject':
        text_input = values['-INPUT-']
        subject_onboarding(text_input)


# Close Window
window.Close()

