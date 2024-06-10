import streamlit as st
import json
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate


uploaded_file = st.file_uploader("Upload Files",type=['json'])

if uploaded_file is None:
    st.stop()

if uploaded_file is not None:
    data = json.load(uploaded_file)
    # st.write(data)


lm = dspy.HFClientVLLM(model="microsoft/Phi-3-medium-128k-instruct", port=38242, url="http://localhost", max_tokens=200)

dspy.settings.configure(lm=lm, trace=[], temperature=0.7)

class GenerateTweet(dspy.Signature):
    """Generate a tweet using a reference text. The tweet text should be technical, informative, accurate, and provide context of what the tweet is even talking about. The tweet should not be like an ad, so it shouldn't sound like someone is trying to convince you to use it. Should have more of a scientific, curious tone."""

    reference_text = dspy.InputField()
    tweet_text = dspy.OutputField(desc="Tweet that is generated based on the reference text.")

class TweetGenerationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = GenerateTweet
        self.predictor_cot  = dspy.ChainOfThought(self.signature)

    def forward(self, reference_text):
        predictions = []
        result = self.predictor_cot(reference_text=reference_text)
        tweet_text = result.tweet_text.split('---')[0].strip()

        return dspy.Prediction(
            reference_text=reference_text,
            tweet_text=tweet_text
        )

def load_dataset(json_data):
    dataset = []
    for item in json_data:
        dataset.append(
                dspy.Example(**item).with_inputs(item.keys())
            )
    return dataset

tweet_generator = TweetGenerationModule()
output = tweet_generator('test')
st.write(output.tweet_text)



