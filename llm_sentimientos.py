# Liberías
from transformers import pipeline

# Inicializa el modelo para análisis de sentimientos
sentiment_analizer = pipeline("sentiment-analysis",
                              model="distilbert-base-uncased")

# Mensajes
messages = ["Amo usar Python para ciencia de datos, es muy divertido",
            "La película era muy aburrida y no se entendía su argumento",
            "Es una buena idea estudiar por las mañanas porque el ambiente es tranquilo",
            "El calor me está matando"]

# Procesamiento
results = []
for message in messages:
    sentiment = sentiment_analizer(message)[0] # análisis de sentimiento
    
    results.append({
        "Message" : message,
        "Sentiment": sentiment["label"]
    })

# Resultados
for result in results:
    print(f"Message: {result['Message']}")
    print(f"Sentiment: {result['Sentiment']}")