# mperf-activity-recognition
Applies CNN based Activity recognition model from public datasets to mperf (400,000+ hours, 400+participants).
mperf data is stored in Spark 3 Dataframes and the activity model is trained in keras.

1. Compute the magnitude
2. Resample the data to conform to training dataset frequency
3. Window the data and detect the activity
4. Smooth the predictions


