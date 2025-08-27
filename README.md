from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. Lista de frases
frases = [
    "Quais são as opções de passagens para Paris?",
    "Gostaria de reservar um hotel no centro.",
    "Quais passeios estão disponíveis em Roma?",
    "Onde posso encontrar bons restaurantes?",
    "Tem promoções de passagens para o Rio?",
    "Quais hotéis têm piscina?",
    "Quais passeios incluem guia turístico?",
    "Restaurantes com comida típica local?",
    "Passagens com desconto para estudantes?",
    "Hospedagem próxima à praia?",
    "Passeios de barco estão disponíveis?",
    "Restaurantes abertos até tarde?"
]

# 2. Vetorização das frases
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(frases)

# 3. KMeans com 4 clusters (um para cada tema)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 4. Imprimir cluster de cada frase
for i, frase in enumerate(frases):
    print(f"Frase: '{frase}' pertence ao cluster {kmeans.labels_[i]}")


