from smartnlp.simililarity.siamese_similarity import SiameseSimilarity

if __name__ == '__main__':
    siamese = SiameseSimilarity('model/', 'model/config.pkl', train=True, data_path='../tutorials/03.poem_generation/data',
                                embedding_file='../GoogleNews-vectors-negative300.bin.gz')
    print(siamese.predict('Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?', 'I m a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?'))
