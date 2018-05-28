import matplotlib.pyplot as plt  
import numpy as np  
  
  
# draw the picture  
def draw(Curve_one, Curve_two, Curve_three, Curve_four):  
  
    plt.figure()  
      
    plot1, = plt.plot(Curve_one[0], Curve_one[1],     'b-', linewidth=1.0, markersize=5.0)  
    plot2, = plt.plot(Curve_two[0], Curve_two[1],     'r-', linewidth=1.0, markersize=5.0)  
    #plot3, = plt.plot(Curve_three[0], Curve_three[1], 'rh-', linewidth=1.0, markersize=5.0)  
    #plot4, = plt.plot(Curve_four[0], Curve_four[1],   'k^-', linewidth=2.0, markersize=10.0)  
      
    # set X axis  
    plt.xlim( [0, 1.03] )  
    plt.xticks( np.linspace(0, 1.0, 11) )  
    plt.xlabel("Recall", fontsize="x-large")  
      
    # set Y axis  
    plt.ylim( [0, 1.03] )  
    plt.yticks( np.linspace(0, 1.0, 11) )  
    plt.ylabel("Precision",    fontsize="x-large")  
      
    # set figure information  
    plt.title("Precision-Recall curve of SSD400 network", fontsize="x-large")  
    plt.legend([plot1, plot2], ("OriginalSSD", "OurSSD"), loc="lower left", numpoints=1)  
    plt.grid(False)  
  
    # draw the chart  
    plt.show()  
  
  
# main function  
def main():  
    # Curve one  
    Curve_one = [ (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),   
                  (1, 0.995, 0.995, 0.995, 0.995, 0.993, 0.993, 0.991, 0.989, 0.989, 0.982, 0.977, 0.967, 0.959, 0.952, 0.941, 0.922, 0.897, 0.847, 0.586, 0) ]  
  
    # Curve two  
    Curve_two  = [ (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),   
                   (1, 1, 1, 1, 1, 1, 1, 1, 1, 0.998, 0.998, 0.992, 0.987, 0.981, 0.977, 0.969, 0.963, 0.936, 0.897, 0.727, 0) ]  
  
    # Curve three  
    Curve_three = [ (0.997050, 0.992867, 0.987220, 0.977600, 0.957650, 0.912350, 0.833100, 0.743753, 0.664700, 0.598011, 0.542317),  
                    (0.103480, 0.239338, 0.411086, 0.578412, 0.734205, 0.858531, 0.928235, 0.956837, 0.979437, 0.984903, 0.997224) ]  
  
    # Curve four  
    Curve_four = [ (0.995750, 0.982433, 0.960270, 0.928964, 0.887517, 0.835059, 0.775115, 0.713273, 0.652159, 0.594189, 0.541702),  
                   (0.107366, 0.258592, 0.421263, 0.570542, 0.700825, 0.805936, 0.884097, 0.938723, 0.972731, 0.990533, 0.998092) ]  
  
    # Call the draw function  
    draw(Curve_one, Curve_two, Curve_three, Curve_four)  
  
  
# function entrance  
if __name__ == "__main__":  
    main()
