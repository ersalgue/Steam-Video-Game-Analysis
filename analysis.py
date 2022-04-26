from wordcloud import WordCloud
import matplotlib.pyplot as plt


def gen_wordCloud(words,imgname,openInTerminal=False):
    wordcloud = WordCloud().generate(' '.join(words))
    
    if (openInTerminal) :
        # Display the generated image:
        plt.figure( figsize=(20,10) )
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    wordcloud.to_file("img/"+imgname+".png")