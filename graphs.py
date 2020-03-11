import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


def graph1():
    #test data
    NB_TFIDF=[0.6194054593589426,0.6148412429593126,0.6669242099384219]
    NB_Count=[0.6839042253059384,0.6909896664549959,0.6562907906685594]

    SVC_TFIDF=[0.7151218156761084,0.7119918590795162,0.6613073378401393]
    SVC_COUNT=[0.7092815417357563,0.6861627147319167,0.5276237674727986]

    LR_TFIDF=[0.7201470820770772,0.694371966021037,0.6857818476125339]
    LR_COUNT=[ 0.7033533408861439,0.7261754895468503,0.6338325600520534]

    labels=['raw_data','after_data_processing','Normalized_data']

    data=pd.DataFrame({"NB_tfidf":NB_TFIDF,"NB_count_vectorizer":NB_Count,"SVC_tfidf":SVC_TFIDF,"SVC_count_vectorizer":SVC_COUNT,"LR_tfidf":LR_TFIDF,"LR_count_vectorizer":LR_COUNT},index=labels)

    #['Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'jet', 'nipy_spectral', 'ocean', 'pink', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'terrain', 'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'Blues_r', 'BrBG_r', 'BuGn_r', 'BuPu_r', 'CMRmap_r', 'GnBu_r', 'Greens_r', 'Greys_r', 'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r', 'PuBuGn_r', 'PuOr_r', 'PuRd_r', 'Purples_r', 'RdBu_r', 'RdGy_r', 'RdPu_r', 'RdYlBu_r', 'RdYlGn_r', 'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r', 'YlOrRd_r', 'afmhot_r', 'autumn_r', 'binary_r', 'bone_r', 'brg_r', 'bwr_r', 'cool_r', 'coolwarm_r', 'copper_r', 'cubehelix_r', 'flag_r', 'gist_earth_r', 'gist_gray_r', 'gist_heat_r', 'gist_ncar_r', 'gist_rainbow_r', 'gist_stern_r', 'gist_yarg_r', 'gnuplot_r', 'gnuplot2_r', 'gray_r', 'hot_r', 'hsv_r', 'jet_r', 'nipy_spectral_r', 'ocean_r', 'pink_r', 'prism_r', 'rainbow_r', 'seismic_r', 'spring_r', 'summer_r', 'terrain_r', 'winter_r', 'Accent_r', 'Dark2_r', 'Paired_r', 'Pastel1_r', 'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r', 'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r', 'magma', 'magma_r', 'inferno', 'inferno_r', 'plasma', 'plasma_r', 'viridis', 'viridis_r', 'cividis', 'cividis_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'rocket', 'rocket_r', 'mako', 'mako_r', 'vlag', 'vlag_r', 'icefire', 'icefire_r']
    ax=data.plot(kind='bar',width=0.35,cmap = cm.get_cmap("PuBu"))
    for p in ax.patches:
        print(p)
    plt.title("Plot of F1-scores")
    plt.ylabel("F1-scores")
    plt.xticks(rotation=0,size=10,horizontalalignment="center")
    plt.legend(loc='upper right', bbox_to_anchor=(0.4, 0.8))
    plt.show()

def graph_2():
    train_NB_TFIDF=[0.9854202382188012]
    test_NB_TFIDF=[0.6669242099384219]

    train_NB_Count=[0.985816544379247]
    test_NB_Count=[0.6562907906685594]

    train_SVC_TFIDF=[0.9875007077212535]
    test_SVC_TFIDF=[0.6552235744581603]

    train_SVC_COUNT=[0.9875007077212535]
    test_SVC_COUNT=[0.5506646693212842]

    train_LR_TFIDF=[0.9875007077212535]
    test_LR_TFIDF=[0.6857818476125339]

    train_LR_COUNT=[ 0.9875007077212535]
    test_LR_COUNT=[0.6338325600520534]

    #################################################
    F_train_NB_TFIDF=[0.6867297562449856]
    F_test_NB_TFIDF=[0.6654012402371308]

    F_train_NB_Count=[0.6875788699625316]
    F_test_NB_Count=[0.6743640715486124]

    F_train_SVC_TFIDF=[0.6815884168474832]
    F_test_SVC_TFIDF=[0.6586025884653977]


    F_train_SVC_COUNT=[0.6838884848972191]
    F_test_SVC_COUNT=[0.6556902231502448]

    F_train_LR_TFIDF=[0.6861881458629645]
    F_test_LR_TFIDF=[0.6527394074561247]

    F_train_LR_COUNT=[0.6886204304858646]
    F_test_LR_COUNT=[0.660611095725471]


    labels=['Naive_Bayes_TFIDF','Naive_Bayes_Count','Linear_SVC_TFIDF','Linear_SVC_Count','Logistic_regression_TFIDF','Logistic_regression_Count']


    data=pd.DataFrame({"Train_NB":train_NB_TFIDF,"Test_NB":test_NB_TFIDF,\
                       "Train_NB_count":train_NB_Count,"Test_NB_count":test_NB_Count,\
                       "Train_SVC_Tfidf":train_SVC_TFIDF,"Test_SVC_Tfidf":test_SVC_TFIDF,\
                       "Train_SVC_count":train_SVC_COUNT,"Test_SVC_count":test_SVC_COUNT,\
                       "Train_LR_Tfidf":train_LR_TFIDF,"Test_LR_Tfidf":test_LR_TFIDF,\
                       "Train_LR_count":train_LR_COUNT,"Test_LR_count":test_LR_COUNT},index=labels)

    #['Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'jet', 'nipy_spectral', 'ocean', 'pink', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'terrain', 'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'Blues_r', 'BrBG_r', 'BuGn_r', 'BuPu_r', 'CMRmap_r', 'GnBu_r', 'Greens_r', 'Greys_r', 'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r', 'PuBuGn_r', 'PuOr_r', 'PuRd_r', 'Purples_r', 'RdBu_r', 'RdGy_r', 'RdPu_r', 'RdYlBu_r', 'RdYlGn_r', 'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r', 'YlOrRd_r', 'afmhot_r', 'autumn_r', 'binary_r', 'bone_r', 'brg_r', 'bwr_r', 'cool_r', 'coolwarm_r', 'copper_r', 'cubehelix_r', 'flag_r', 'gist_earth_r', 'gist_gray_r', 'gist_heat_r', 'gist_ncar_r', 'gist_rainbow_r', 'gist_stern_r', 'gist_yarg_r', 'gnuplot_r', 'gnuplot2_r', 'gray_r', 'hot_r', 'hsv_r', 'jet_r', 'nipy_spectral_r', 'ocean_r', 'pink_r', 'prism_r', 'rainbow_r', 'seismic_r', 'spring_r', 'summer_r', 'terrain_r', 'winter_r', 'Accent_r', 'Dark2_r', 'Paired_r', 'Pastel1_r', 'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r', 'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r', 'magma', 'magma_r', 'inferno', 'inferno_r', 'plasma', 'plasma_r', 'viridis', 'viridis_r', 'cividis', 'cividis_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'rocket', 'rocket_r', 'mako', 'mako_r', 'vlag', 'vlag_r', 'icefire', 'icefire_r']
    ax=data.plot(kind='bar',width=0.35,cmap = cm.get_cmap("accent"))
    for p in ax.patches:
        print(p)
    plt.title("Plot of F1-scores")
    plt.ylabel("F1-scores")
    plt.xticks(rotation=0,size=10,horizontalalignment="center")
    plt.legend(loc='upper right', bbox_to_anchor=(0.4, 0.8))
    plt.show()

def train_test_accuracy_n_k():

    K_best=[1,21,41,61,81,101]
    #chi-square
    #NB

    c_Train_score=[0.67288900432421, 0.6929559896967233, 0.716147242917972, 0.7316252839511791, 0.7434135753565445, 0.7535431931240054]
    c_Test_score=[0.67289442371811, 0.6780606211369762, 0.6833936653569009, 0.686547186435547, 0.6883713882914673, 0.6890430646506027]

    [0.67288900432421, 0.7217298311441064, 0.7492169814165078, 0.7696690103068501]
    [0.67289442371811, 0.6849571880738206, 0.687438607531307, 0.6884427383723563]
    [1, 51, 101, 151]

    #SVC
    Tr=[0.8224503400028962]
    tes=[0.7067420023409579]
    #logistic regression
    # Train_score=[0.682003199885289, 0.7097113923934769, 0.7346145260132694, 0.7505731970977011, 0.7624751068584278,0.7522267699440927]
    # Test_score=[0.6761365685726728, 0.6809736860996904, 0.6910870765039677, 0.696529677843569, 0.6994157087562588,0.6930643016084218]






    #mutual info

    #NB
    m_Train_score = [0.67288900432421, 0.7049862800392988, 0.7172584451479878, 0.7248612052702235, 0.7305627500528837,0.7356427473456812]
    m_Test_score = [0.67289442371811, 0.6888080408960657, 0.6951529380386403, 0.6960073430260855, 0.6958721165201084,0.6964071043040513]

    #SVC
    0.7698736096938236
    0.6718576684924009
    #LR
    # Train_score =[0.5938625569723783, 0.647643892693267, 0.6728810266280256, 0.6902352753378448, 0.7032904211521269, 0.7141177412420636]
    # Test_score =[0.5340627015800813, 0.6057869661315878, 0.6274904899558379, 0.6378266991425863, 0.6433150283151076, 0.6479845126095768]




    plt.plot(K_best,c_Train_score,'bo--',label="Train_score-chi2")
    plt.plot(K_best, c_Test_score, 'b+--', label="Test_score-chi2")
    plt.plot(K_best,m_Train_score,'go--',label="Train_score-mutal_info")
    plt.plot(K_best,m_Test_score, 'g+--', label="Test_score-mutual_info")

    plt.title("Train vs Test F1-Score k_best features on Naive Bayes")
    plt.xlabel("K_best features")
    plt.ylabel("F1-scores")
    plt.legend()
    plt.show()

train_test_accuracy_n_k()