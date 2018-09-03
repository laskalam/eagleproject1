import matplotlib.pyplot as plt
import seaborn as sns


def display_outlier(pd, feature):
    fig, ax = plt.subplots()
    ax.scatter(x = pd[feature], y = pd['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel(feature, fontsize=13)
    plt.show()


def display_distrib(pd, feature):
    plt.figure()
    sns.distplot(pd[feature].dropna() , fit=norm);
    (mu, sigma) = norm.fit(pd[feature].dropna())
    
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    plt.show()
