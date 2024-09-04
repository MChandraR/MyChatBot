import json
import random
import voicevox
import torch 

import asyncio
import romajitable
import winsound
import numpy as np
from voicevox import Client
from neural_n import NeuralNet
from nplUtil import tokenize,stemm,bag

async def speak(words):
    async with Client() as client:
        audio_query = await client.create_audio_query(
            words,
            speaker=8
        )
        
        with open("voice.wav","wb") as f:
            f.write(await audio_query.synthesis(speaker=8))
        
        winsound.PlaySound("voice.wav",winsound.SND_FILENAME)
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("dataset/model.json","r") as data:
     intents = json.load(data)

FILE = "dataset/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()
ignored_char = ["?","!",".",","," "];

bot_name = "Mia"

async def main():
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        sentence = [stemm(w) for w in sentence if w not in ignored_char]
        X = bag(all_words,sentence)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() >= 0.75:
            for intent in intents:
                if tag == str(intent):
                    datas = []
                    for data in intents[str(intent)]:
                        datas.append(data['response'])
                    res = random.choice(datas)
                    jpres = romajitable.to_kana(res).hiragana
                    jpress = ""
                    for w in jpres:
                        if w == "・":
                            w = ""
                        jpress += w
                    
                    print(f"{bot_name}: {res}\n",jpress)
                    await speak(jpress)
                    
        else:
            print(f"{bot_name}: Saya tida mengerti...")

if __name__ == "__main__":
    asyncio.run(main()) 