import streamlit as st
import json
import dspy
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot
from dspy.evaluate.evaluate import Evaluate


st.set_page_config(layout="wide")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Files",type=['json'])

if uploaded_file is not None:
    json_data = json.load(uploaded_file)

if uploaded_file is None:
    if False:
        st.stop()
    else:
        test_file = 'test_data/dspy_assert_texts.json'
        with open(test_file) as f:
            json_data = json.load(f)



lm = dspy.HFClientVLLM(model="microsoft/Phi-3-medium-128k-instruct", port=38242, url="http://localhost", max_tokens=200)

dspy.settings.configure(lm=lm, trace=[], temperature=0.7)

def initialize_session_state():
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
        st.session_state.traindata_selected = []

class GenerateTweet(dspy.Signature):
    """Generate a tweet using a reference text. The tweet text should be technical, informative, accurate, and provide context of what the tweet is even talking about. The tweet should not be like an ad, so it shouldn't sound like someone is trying to convince you to use it. Should have more of a scientific, curious tone."""

    source_text = dspy.InputField()
    tweet_text = dspy.OutputField(desc="Tweet that is generated based on the reference text.")

class TweetGenerationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = GenerateTweet
        self.predictor_cot  = dspy.ChainOfThought(self.signature)

    # def forward(self, reference_text):
    # def forward(self, **kwargs):
    def forward(self, source_text):
        predictions = []
        # result = self.predictor_cot(reference_text=reference_text)
        # result = self.predictor_cot(**kwargs)
        result = self.predictor_cot(source_text=source_text)
        tweet_text = result.tweet_text.split('---')[0].strip()

        return dspy.Prediction(
            # reference_text=reference_text,
            source_text=source_text,
            tweet_text=tweet_text
        )

def load_dataset(json_data):
    dataset = []
    for item in json_data:
        # print("item.keys():", item.keys())
        print("list(item.keys()):", list(item.keys()))
        dataset.append(
                # dspy.Example(**item['input']).with_inputs(*list(item.keys()))
                dspy.Example(**item['input']).with_inputs('source_text')
            )
    return dataset

def interactive_metric(gold, pred, trace=None, return_individual_scores=False):
    # st.write(pred)
    st.session_state.predictions.append(pred)
    # if st.button('Accept'):
        # return True
    
    # if st.button('Reject'):
        # return False
    return False


def add_to_trainset(prediction, i):
    # st.write(prediction)
    example = dspy.Example(**prediction).with_inputs('source_text')
    # st.write('Example:')
    # st.write(example)
    st.session_state.predictions[i] = None
    st.session_state.traindata_selected.append(example)


def remove_from_trainset(i):
    del st.session_state.traindata_selected[i]


@st.experimental_fragment(run_every=1)
def display_trainset():
    # st.write('test')
    # st.write(st.session_state.traindata_selected)
    if st.session_state.traindata_selected:
        keys = st.session_state.traindata_selected[0].keys()
        for i, example in enumerate(st.session_state.traindata_selected):
            # keys = example.keys()
            # cols = st.columns(2)
            cols = st.columns(len(keys) + 1)
            # with cols[0]:
                # st.write(example)
            with cols[0]:
                st.button('Remove', on_click=remove_from_trainset, args=(i,), key=i)
            for j, key in enumerate(keys):
                with cols[j+1]:
                    st.write(example[key])
            # with cols[1]:



@st.experimental_fragment(run_every=1)
def display_predictions():
    # predictions = st.session_state.predictions.copy()
    predictions = st.session_state.predictions
    # for prediction in st.session_state.predictions:
    keys = []
    for pred_i, prediction in enumerate(predictions):
        if prediction:
            keys = prediction.keys()
            break

    cols = st.columns([0.3] + [1]*len(keys))
    with cols[0]:
        st.write('Predictions')
    for i, key in enumerate(keys):
        with cols[i+1]:
            st.write(key)

    '---'

    if predictions:
        # keys = predictions[0].keys()
        for pred_i, prediction in enumerate(predictions):
            if not prediction:
                continue
            # cols = st.columns(2)
            # cols = st.columns(len(keys) + 1)
            # cols = st.columns([0.3, 1, 1])


            cols = st.columns([0.3] + [1]*len(keys))
            # with cols[0]:
                # # st.write(st.session_state.predictions)
                # # st.write(prediction)
                # st.code(prediction)
                # st.write(prediction.tweet_text)
            with cols[0]:
                st.button('Add to trainset', on_click=add_to_trainset, args=(prediction, pred_i), key=f'add_{pred_i}')
            for i, key in enumerate(keys):
                with cols[i+1]:
                    st.write(prediction[key])
            '---'


def run_evaluation():
    st.session_state.predictions = []
    dataset = st.session_state.traindata_selected + dataset_imported 

    teleprompter = LabeledFewShot()
    compiled_program = teleprompter.compile(tweet_generator, trainset=dataset)

    evaluate = Evaluate(metric=interactive_metric, devset=dataset, num_threads=num_threads, display_progress=True, display_table=5)
    # if st.session_state.traindata_selected != []:
        # evaluate(compiled_program)
    # else:
        # evaluate(tweet_generator)

    # evaluate(tweet_generator)
    evaluate(compiled_program)

initialize_session_state()
'### Accepted Outputs'
display_trainset()
'---'
'---'
# '### Predictions'
display_predictions()

st.button('Run Evaluation', on_click=run_evaluation)
with st.sidebar:
    st.button('Run Evaluation', on_click=run_evaluation, key='sidebar_run_evaluation')


tweet_generator = TweetGenerationModule()
# output = tweet_generator('test')
# st.write(output.tweet_text)
dataset = load_dataset(json_data)
print("dataset:", dataset)
dataset_2 = [
    dspy.Example(
        source_text="The quick brown fox jumps over the lazy dog."
    ).with_inputs("source_text"),
    dspy.Example(
        source_text="The slow brown fox jumps over the lazy dog."
    ).with_inputs("source_text"),
]
# st.write(dataset)
# st.write(dataset_2)

# dataset_imported = dataset_2
dataset_imported = dataset


from dspy.evaluate.evaluate import Evaluate

num_threads = 1
# evaluate = Evaluate(metric=interactive_metric, devset=dataset_2, num_threads=num_threads, display_progress=True, display_table=5)
# evaluate(tweet_generator)


