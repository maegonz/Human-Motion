import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BLEUCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, Y_val, tokenizer):
        super().__init__()
        self.X_val = X_val
        self.Y_val = Y_val
        self.tokenizer = tokenizer
        self.smoothie = SmoothingFunction().method1  # Évite un BLEU = 0 sur des phrases courtes

    def on_epoch_end(self, epoch, logs=None):
        bleu_scores = []
        for i, motion in enumerate(self.X_val):
            pred_sentence = self.generate_text(motion)
            ref_sentence = self.tokenizer.sequences_to_texts([self.Y_val[i]])[0]

            bleu = sentence_bleu([ref_sentence.split()], pred_sentence.split(), smoothing_function=self.smoothie)
            bleu_scores.append(bleu)

            # Affichage à chaque prédiction
            print(f"🎯 Référence: {ref_sentence}")
            print(f"🤖 Prédiction: {pred_sentence}")
            print(f"🔵 BLEU Score: {bleu:.4f}\n{'-'*40}")

        avg_bleu = np.mean(bleu_scores)
        print(f"🔹 BLEU moyen après epoch {epoch+1}: {avg_bleu:.4f}")

    def generate_text(self, motion):
        input_seq = np.expand_dims(motion, axis=0)  # Ajouter une dimension pour batch

        # Créer un modèle d'encodeur pour obtenir state_h et state_c
        encoder_model = Model(inputs=self.model.input, outputs=[self.model.layers[1].output, self.model.layers[2].output])
        state_h, state_c = encoder_model.predict(input_seq)

        # Initialisation du séquenceur cible
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.tokenizer.word_index['startseq']

        decoded_sentence = []
        for _ in range(20):  # max_len (limité ici à 20 mots)
            # Décoder la séquence en utilisant l'état de l'encodeur
            output_tokens, state_h, state_c = self.model.layers[3](target_seq, initial_state=[state_h, state_c])
            
            # Sélectionner le mot avec la plus grande probabilité
            word_idx = np.argmax(output_tokens[0, -1, :])
            word = self.tokenizer.index_word.get(word_idx, '')

            # Arrêter si l'on atteint 'endseq' ou un mot inconnu
            if word == 'endseq' or word == '':
                break

            decoded_sentence.append(word)
            target_seq[0, 0] = word_idx  # Mise à jour du séquenceur cible

        return ' '.join(decoded_sentence)