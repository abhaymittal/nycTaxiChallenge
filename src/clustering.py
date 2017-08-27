import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import re
try:
    import cPickle as p
except:
    import Pickle as p
    
def create_line_plot(labels, x_values, y_values, y_label, x_label, title):
    """
    Creating a line plot from the given values and saving it in a file
    :param labels:
    :param x_values:
    :param y_values:
    :param y_label:
    :param x_label:
    :param title:
    :return:
    """
    if len(labels) != len(x_values):
        print "Inconsistent dimensions of lables and values!!!"
        return
    plt.close('all')
    plt.plot(x_values, y_values, 'or-', linewidth=3)  # Plot the first series in red with circle marker
    # This plots the data
    plt.grid(True)  # Turn the grid on
    plt.ylabel(y_label)  # Y-axis label
    plt.xlabel(x_label)  # X-axis label
    plt.title(title)  # Plot title
    plt.xlim(np.min(x_values), np.max(x_values))  # set x axis range
    plt.ylim(np.min(y_values), np.max(y_values))  # Set yaxis range
    plt.gca().set_xticks(x_values)  # label locations
    plt.gca().set_xticklabels(labels)  # label values
    # Save the chart
    plt.savefig("./line_plot_" + re.sub(" ", "_", title.lower()) + ".jpg")
    return


def calculate_k_means_variance(df,num_clusters):
    print "Clustering with ",num_clusters,"clusters"
    kmeans = KMeans(n_clusters=num_clusters,n_jobs=10).fit(df)
    return kmeans.inertia_ # sum of squared distances to the closest centroid for all observations in the training set


def plot_variance(df,n_clusters):
    y_array=[]
    for num_clusters in n_clusters:
        y_array.append(calculate_k_means_variance(df,num_clusters))
    print "SSE = ",y_array
    x_labels = map(lambda x : str(x),n_clusters)
    create_line_plot(labels=x_labels,x_values=n_clusters,y_values=y_array,y_label="Total Within Cluster Variance (Log Scale)",x_label="Number of clusters",title="Within Cluster Variance for different clusters")



if __name__ == '__main__':
    dataFrame = pd.read_csv('../data/train.csv')
    print "Filter data based on duration and coordinates"
    dataFrame=dataFrame[dataFrame['trip_duration']>=60]
    dataFrame =dataFrame[dataFrame['trip_duration']<1939736]
    long_limit=[-74.257159, -73.699215]
    lat_limit=[40.495992, 40.915568]
    dataFrame=dataFrame[(dataFrame['pickup_longitude']>=long_limit[0])&(dataFrame['pickup_longitude']<=long_limit[1])]
    dataFrame=dataFrame[(dataFrame['pickup_latitude']>=lat_limit[0])&(dataFrame['pickup_latitude']<=lat_limit[1])]
    dataFrame=dataFrame[(dataFrame['dropoff_longitude']>=long_limit[0])&(dataFrame['dropoff_longitude']<=long_limit[1])]
    dataFrame=dataFrame[(dataFrame['dropoff_latitude']>=lat_limit[0])&(dataFrame['dropoff_latitude']<=lat_limit[1])]
    print "Prepare data for clustering"
    dataFrameProc=dataFrame.copy()
    pdtime=pd.to_datetime(dataFrameProc.pickup_datetime)
    ddtime=pd.to_datetime(dataFrameProc.dropoff_datetime)
    del dataFrameProc['pickup_datetime']
    del dataFrameProc['dropoff_datetime']
    dataFrameProc['pickup_date']=pdtime.dt.date
    dataFrameProc['pickup_time']=pdtime.dt.time
    dataFrameProc['pickup_dow']=pdtime.dt.dayofweek # Pickup day of week
    dataFrameProc['pickup_hour']=pdtime.dt.hour
    dataFrameProc['dropoff_date']=ddtime.dt.date
    dataFrameProc['dropoff_time']=ddtime.dt.time
    dataFrameProc['dropoff_dow']=ddtime.dt.dayofweek #Dropoff day of week
    dataFrameProc['dropoff_hour']=ddtime.dt.hour
    lat=pd.Series(dataFrameProc['pickup_latitude'])
    longt=pd.Series(dataFrameProc['pickup_longitude'])
    lat=lat.append(dataFrameProc['dropoff_latitude'])
    longt=longt.append(dataFrameProc['dropoff_longitude'])
    df=longt.to_frame(name='longitude')
    df['latitude']=lat
    # print "Determine number of clusters"
    # plot_variance(df,np.arange(10,20))
    print "Run KMeans"
    kmeans=KMeans(n_clusters=16,n_jobs=10).fit(df)
    p.dump(kmeans,open('./kmeansDump.p','wb'))
