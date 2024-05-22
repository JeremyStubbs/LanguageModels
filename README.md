**Large Language Models**

They evolved from translation models. It is helpful to consider what they evolved from to estimate the current architecture. The initial structure was an encoder-decoder. The encoder created word embedding tensors also called a context vector. These explained the positional relationship between words in a sentence. It is still helpful to think of large language models in terms of two parts, but I would rename them into the reading comprehension part and response production part, although both tasks may be part of the same model and the architecture that accomplishes both tasks may overlap.

It does not operate on Boolean logic (although the transistor does) but rather statistics - groups of neurons perform complex statistical operations on matrices not Boolean mathematics on 0s and 1s. Chatgpt told me this and Geoffrey Hinton is famous partly for modeling networks as Boltzman fields (stats stuff). The statistics determine 1) most likely meaning of a word or phrase and 2) how to craft a response.

It is also helpful to understand how the human mind works. If I tell you a story, you will (assuming average intelligence) be able to draw conclusions about events in it and make predictions about it. Much of this revolves around creating a visualization of the story. Your mind has much practice in visualizing things that happen on earth and assigning all the appropriate physics to it (time, gravity, size and speed of things). Your brain works on rather simple neurons - threshold activation. It accomplishes this task by using extremely complex networks. I imagine the brain as a humming mass of web-like electricity. You cannot make these sorts of 3D connections on silicon, but you don’t need to because silicon neurons can do extremely complex matrix/statistical operations. But even with silicon there will be some similarity of network complexity at least functionally to accomplish these tasks.

Side note: computer AI models have great difficulty understanding extremely long inputs - I doubt it could read a book in one sitting and make sense of it. It could break it up into parts and make sense of them together. The human mind on the other hand could easily do that. Also AI models are extremely specialized: a language model cannot interpret videos and pictures (it is different code running on a different gpu). Lastly, generative AI is simply a model that writes code to create pictures/videos. AGI is simply a collection of different models.

At the most basic level words are passed in one by one and used to create a humming mass of electricity that encompasses all their relationships. The structures that make this possible are **programmable memory units called long-short term memory networks and transformer networks**. (Linear, relu, and dense neurons have supporting roles.) LSTM and transformers have the ability to form memories, which are essential for understanding the relationships between everything. Silicon differs from biology in that memory is stored as matrices/tensors - even the definitions of words are stored as tensors. As the words are passed into the network, the values of the matrices change. These matrices are analogous to the context vectors of early models, but now they must encompass a lot more information: content comprehension and style directives of the response (length, writing style, etc).

As I mentioned, the first task is reading comprehension. Most likely all the words are processed before the response production part starts spitting stuff out. I believe that the architecture is massively parallel input networks. There are many many parallel composite networks built from LSTM and transformer networks together with other neurons. Each receives every input word. These networks are highly connected to each other. This first part of the model “programs” or “primes” the second part of the model to create a “world” that has all the expected dependencies and relationships of the response using some sort of flexible statistics that can take any input. How these networks are connected to the response production part of the model is a mystery to me. For all I know it may change based on the input.

The response production part makes words one by one. It ensures flow and completion through feedforward and feedback mechanisms. Sentences are built one word at a time (I don’t think it’s physically possible to think of whole paragraphs at a time). Paragraphs are just related sentences and stories are composed of related paragraphs. Let’s think more about it conceptually before talking about architecture. Each sentence has a symbolic purpose: description, action, fact, etc. Together they create a paragraph which itself has a symbolic purpose. The functionality of building complex related symbols is similar to building sentences that serve specific symbolic purposes. The difference is the symbols are based on past symbols! This reflects actual writing: “The last sentence said — so —. The last paragraph said — so —. The whole story up to here says — and I want to say — so —. Stylistically, I want suspense before the climax so I will start with — then do —”. But we will get there.

The concrete details are explained in the foundational papers for language models describing feedback at each time step (after each output word): the following things are sent back into the model: the previous time step’s output (matrix representation of the word), the model's weights (in a matrix), and the context tensor for each input word.

In the first part of the model, the statistical operations appropriately alter the values of the short term/programmable memory units matrices and the adjustable weights. In the second part of the model, the statistical operations select which information to pull from long-term memory (database) and predict the most likely resultant matrices which in turn predict an appropriate next word.

Here’s the last thing I will say. The flow of the response is coded in the weights of the model. So are the style variations. There are an unlimited number of ways to construct a sentence (noun or noun equivalent first in most, verb first in some commands, etc). You are not going to have a unique subnetwork for every kind of sentence; the network is redundant and variation is accomplished by changing the weights. These weights are determined by the input sequence and included in the matrices passed through the model.

The exact architecture now is a mystery because there’s still money to be made. You may elucidate the architecture by creating a simple world with a limited number of words: a world of two people who live in different homes and go to the same grocery store to buy food (one item), work on the weekdays at the same place and on weekends play together in the park. Experiment with LSTM and transformer architectures and train the model to give definitions and make simple responses to simple queries.

**Initial musings:**

When I first started this I had two maxims: 1. The combinations of words exceed the number of neurons. 2. All words can be defined with other words. I’m not so sure that the memory of definitions actually uses other words (cyclical definitions since they all define each other). The matrices can, if they are sufficiently large, encompass the definitions well.

Most likely all the words are processed (into matrices) before the response production part starts spitting stuff out (matrices). I previously thought that this might be too complex to convert to a set of matrices, but the truth is if you have a tensor of 100,000 elements and each element can have thousands of different states, you can uniquely describe any input (up to a word limit).

**How to start:**

Train words and their characteristics. 1) Verb, noun, adjective 2) If noun, whether living or inanimate and many other factual details ….

Next create an AI that can answer simple questions with sentences based on the basic grammar rules of English from these words. The model layers must reflect the sentence structure.

**Style:**

There are an unlimited number of ways to construct a response. Style parameters: length of response, writing style/voice, sentence structure (noun or noun equivalent first in most, verb first in some commands, etc). How it does it:

1. The style parameters are passed along with the input parameters.
2. The neurons select weights based on style parameters

Make the model build sentences with different ratings in the following categories: declarative/informational, descriptive, opinion (persuasive and other kinds), poetic/rhythmic, etc.

**More knowledge/connected to the web:**

This is connected to a database which it accesses when the question/response exceeds the limit of the knowledge stored in the model’s memory units. All these values are coded into the database by an entirely separate program (not by humans) which the model accesses as needed.

Eg if the user says “Tell me about a rare cat breed,” it will search the database for information about cats and direct the sentence model to convey the information in a descriptive sentence about those cats.

Eg the user asks “can dogs run?” It looks up dogs. A dog is a land based quadruped. run is a verb that means fast movement of land animals for short periods of time.

**Linguistic Logic:**

Humans often visualize things. Create an imaginary world in our minds eye in which stuff plays out. The AI must do something similar.

Eg a squirrel chitters a high pitched sound. If something else makes that sound it also chitters.

Eg, if you start a sentence with “ running through the forest” it can’t continue “the dolphin jumped out of the water” unless it’s a kelp forest even though it would be grammatically correct. It could also continue correctly, “he tripped over a root” or “she heard a parrot squawk” but not “the concert reached a crescendo”. It is not clear how the information in the database is connected to the sentence writing part but I hypothesize a little bit about it later.

The logical consistency/ factual accuracy depends on the context of the response.

Eg for a question that requires a factual response, a brick cannot fly unless it is thrown, but in a fictional story it can do anything.

Conclusion:

At some point the training goes from human to AI. An AI model must be developed that can ask questions and evaluate answers (not create answers).

First level: train words and their characteristics. 1) Verb, noun, adjective 2) If noun, whether living or inanimate and many other factual details …You start with a model that can “learn” that a dog is a land animal and land animals live on land therefore dogs live on land.

The next step is to create an AI that can answer simple questions with sentences based on the basic grammar rules of English from these words. This can be done simply but it must be nested in a massive program.

Then make it build sentences with different ratings in the following categories: declarative/informational, descriptive, opinion (persuasive and other kinds), poetic/rhythmic, etc.

This is connected to a database which it accesses when exceeds limit of its knowledge. Encyclopedia of knowledge/think every possible google search. All these values are coded into the database by an entirely separate program (not by humans).

The model layers must reflect the sentence structure: a past tense conjugation always goes to the same subnetwork?

There are many training cycles to determine which parameters to vary. The modifications are chosen by a sort of search algorithm (divide and conquer).

Questions for AI:

Ask about symbolism of an entirely new story.

Ask it to identify opinion vs fact.

Ask it to draw connections between two stories.