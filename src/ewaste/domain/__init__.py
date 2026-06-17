"""Domain layer: the pure core of the e-waste scanner.

Modules here describe *what* the application reasons about (detections, hauls,
material content, valuations, impact estimates) and the calculations over them.
This layer has no dependency on Ultralytics, Streamlit, yfinance, the file
system, or the network; everything external is reached through ports.
"""
