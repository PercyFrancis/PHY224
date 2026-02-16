# %% [markdown]
# # PHY224 Circuits : Resistor

# %% [markdown]
# ## Student Information
# - Student Name: Percival Francis
# - Partner Name: Justin Nicholson
# - TA Name: Shaun

# %% [markdown]
# ## Upload data files
# Upload your data file (.csv file) that contains the data you measured and used in the experiment

# %% [markdown]
# ## Question 1
# Write the python code in the next box to load any needed data, fit models to the data, perform any additional calculations, and create the required plots. If you work in a Jupyter notebook you should submit include a PDF rendering of the notebook so that it can be graded. If you work in a Python text file (e.g. in Spyder or VS Code) you should upload the Python file separately to the figures and captions, and use the word file to attach figures and write captions.

# %%
# imports 
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# %% [markdown]
# Measured Resistance of the Resistor using an Ohmmeter:

# %%
MEASURED_RESISTANCE = 106010 # in ohms
MEASURED_ERROR = 300 # in ohms

# %% [markdown]
# Values from CSV file:

# %%
resistor_data = np.loadtxt("./data/resistor_data.csv", delimiter=',', comments='#', unpack=True, skiprows=2) # retrieve from csv.
voltage_data = resistor_data[0] # potential difference
ampere_data = resistor_data[1] # current
ohm_data = resistor_data[4] # resistance

voltage_error = resistor_data[2] # potential difference error
ampere_error = resistor_data[3] # current error
ohm_error = resistor_data[5] # resistance error

# %% [markdown]
# ## Question 2
# Plot the fit to the resistor data with linear axes. In the next question you'll add a caption for this figure. (if you made one file with multiple plots, it's easiest to upload the same file to both questions so that we don't miss anything when grading)
# 

# %% [markdown]
# Function for creating graphs with different cutoffs. Different Cutoffs are used for increasing the visibility of the errors.

# %%
def create_graph(cutoff):
    plt.errorbar(x = range(1 + cutoff, 21), y = ohm_data[cutoff:]/1000, yerr=ohm_error[cutoff:]/1000, fmt= 'o',
               ms=2, color='teal', ecolor='aquamarine', label='Resistance k$\Omega$')

    # measured resistance
    plt.plot(range(1 + cutoff, 21), [MEASURED_RESISTANCE/1000 for _ in range(20 - cutoff)],
            label = "Measured Resistance", color="red")


    plt.plot(range(1 + cutoff, 21), [(MEASURED_RESISTANCE + MEASURED_ERROR)/1000 for _ in range(20 - cutoff)],
            label = "Measured Resistance Error Upper Bound", color="orange", linestyle='--')


    plt.plot(range(1 + cutoff, 21), [(MEASURED_RESISTANCE - MEASURED_ERROR)/1000 for _ in range(20 - cutoff)],
            label = "Measured Resistance Error Lower Bound", color="brown", linestyle='--')

    # Legend and titles
    plt.xticks(range(1 + cutoff, 21))

    plt.legend()
    plt.xlabel('Trial Voltage (V)')
    plt.ylabel('Measured Resistance (k$\Omega$)')
    plt.title('Trial Voltage (V) vs Measured Resistance (k$\Omega$)')

    plt.show()



# %% [markdown]
# All data points (1 to 20)

# %%
create_graph(0)

# %% [markdown]
# Some data points (6 to 20)

# %%
create_graph(5)

# %% [markdown]
# Some data points (11 to 20)

# %%
create_graph(10)

# %% [markdown]
# ## Question 3
# 
# Write a suitable caption that describes the plot. You should include the meaning of each point or line on your plot, any quantities requested (exponent with uncertainty) with appropriate units. Include the regression coefficients.
# 
# A caption needs to explain what’s in the plot and how it relates the data and models. A person with some background in your experiment should be able to read the caption and understand the figure without needing critical information from somewhere else.

# %% [markdown]
# These three plots show the relationship between the voltage of a particular trial, and how it compares to resistance. The voltage chosen for each trial ranges from 1V to 20V. The first plot contains all the information shown in the next two, but the other two plots make it easier to see that resistance values of the higher voltage trials all fall within the error of the resistance measured using the ohmmeter.
# 
# Since in all the trials > 10V have the resistance falling within the value predicted by Ohm's law, it is safe to say that Ohm's law is followed only for higher voltages (greater than 10V).
# 
# The data was obtained from `resistor_data.csv` which was gathered during the lab.
# 
# A more in-depth analysis of the data is shown below:

# %% [markdown]
# The plots shown demonstrate collected data on resistors fitted to linear axes. The plots demonstrate a calculated resistance (kΩ) in relation to each trial voltage (V), for 20 voltage data points. This uses Ohm’s Law in the R=V/I configuration, where V indicates the voltage of different trials (1V to 20V), and I is the associated current, to find resistance. We also see, as we use different y-axis scales to focus on the range of data, there exists degrees of uncertainty lying within an estimated ± ~0.5-1 kΩ range. This is demonstrated by the dashed orange line, indicating the upper bound error in measured resistance and the maroon/burgundy colored dashed line indicating the lower bound estimated error in measured resistance. The red horizontal line indicates the measured resistance across all trials, averaged. This value is ~106.0 kΩ. What can be seen is a relatively constant data set across the various voltage range of trial data points. Also, the green lines indicating the range of resistance decrease as the voltage increases through every testing point. This may indicate the system becomes more efficient over usage. This demonstrates (an imperfect) linear trend- aiding in the assumption that the resistor behaves with Ohmic principles of perfect resistance through the system with proportional voltage and current relationships. 

# %% [markdown]
# ## Question 4
# Add any other details you think are important after this point.

# %%
#Appendix

# %% [markdown]
# All tests are conducted in assumed room temperature environments. The near zero slope on this graph aids in indicating near-constant resistance through the resistor system, seen in the model. All 20 data points are close to the average, perhaps more sensitive measuring tools or measuring philosophies may yield different results in future testing.
# 
# The source of the data is found in the file `resistor_data.csv` located in `/resistor_lightbulb/data/resistor_data.csv`.


