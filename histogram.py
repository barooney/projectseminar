import matplotlib.pyplot as plt


# histogram 
def create_histograms(df, bins_input, name):
    X = df['funny'].tolist()
    # when creating a histogram bins are different: the lower value is INCLUDED while the upper value is excluded
    # for the last bin, both lower AND upper value are included
    max_funny = df['funny'].max()
    #n, bins, patches = plt.hist(X, int(num_bins), facecolor='blue', density=True)
    #n, bins, patches = plt.hist(X, [0,1,2,3,4,5,6,max_funny], facecolor='blue', density=True)
    
    bins_for_histo = [number+1 for number in bins_input] # change bins so that lower value is included but upper value is not
    bins_for_histo[-1] = max_funny
    
    n, bins, patches = plt.hist(X, bins_for_histo, edgecolor='white', density=True)
    # patches[0].set_facecolor('b')   
    # patches[1].set_facecolor('green')
    # patches[2].set_facecolor('yellow')
    # patches[3].set_facecolor('black') 
    # patches[4].set_facecolor('r')
    
    # this must be changed to have different colors
    
    plt.title('histogram')
    plt.xlabel('funny votes')
    plt.ylabel('frequency densitiy')
    #plt.show()
    plt.savefig('./doc/images/density_' + name + '.pdf', format='pdf')
    plt.close()