**Large Language Models**

I wrote this before architecture models were published in journals and Facebook made llama open source. 
They evolved from translation models. The initial structure was an encoder-decoder. The encoder created word embedding tensors also called context vectors. These explained the positional relationship between words in a sentence. It is still helpful to think of large language models in terms of two parts, but I would rename them the reading comprehension stage and response production stage.

The concrete architecture details are explained in the foundational papers for language models. At each time step (after each output word): the following things are fed back into the model: the previous time step’s output (matrix representation of the word), the model's weights (in a matrix), and the context tensor for each input word. I think this is still the general architecture but on a much larger scale.

It does not operate on Boolean logic, mathematics on 0s and 1s (although the transistor does), but rather statistics - groups of neurons perform complex statistical operations on matrices. Chatgpt told me this and Geoffrey Hinton is famous partly for modeling networks as Boltzman fields (stats stuff). The statistics determine 1) most likely meaning of a word or phrase and 2) how to craft a response.

It is helpful to understand how the human mind works. If I tell you a story, you will (assuming average intelligence) be able to draw conclusions about events in it and make predictions about it. Much of this revolves around creating a visualization of the story. Your mind has much practice in visualizing things that happen and assigning all the appropriate physics to it (time, gravity, size and speed of things). Your brain works on rather simple neurons - threshold activation. It accomplishes this task by using extremely complex networks. I imagine the brain as a humming mass of web-like electricity. You cannot make these sorts of 3D connections on silicon, but you don’t need to because silicon neurons can do extremely complex matrix/statistical operations. But even with silicon there will be some similarity of network at least functionally to accomplish these tasks.

The structures that make this possible are programmable memory units called long-short term memory networks and transformer networks. (Linear, relu, and dense neurons have supporting roles.) LSTM and transformers to store matrix values between time steps of the program. These matrices are analogous to the context vectors of early models, but now they must encompass a lot more information: content comprehension and style directives of the response (length, writing style, etc). In summary, words are passed in one by one and used to create matrices that encompass all their relationships.

Side note: computer AI models have great difficulty understanding extremely long inputs - I doubt it could read a book in one sitting and make sense of it. It could break it up into parts and make sense of them together (which may or may not be different). Also AI models are extremely specialized: a language model cannot interpret videos and pictures (it is different code running on a different gpu). AGI is simply a collection of different models. Lastly, generative AI is simply a model that writes code to create pictures/videos. 

As I mentioned, the first task is reading comprehension done by the first stage. The first stage makes sense of the prompt and provides directives to the second stage. Most likely all the words are processed by the first stage before the response production starts by the second stage. The first stage then passes matrix(ces) and weights to the second stage which tell it how to begin the response. 

The response production stage makes sentences one word at a time (I don’t think it’s physically possible to think of whole paragraphs at a time). It ensures flow and completion through feedforward and feedback mechanisms with both itself and the first stage. There is constant evaluation of what the second stage is doing and readjustments of the matrices and weights of the second stage to accomplish its directives. 

I believe that the architecture of the first stage of the model is massively parallel input networks. There are many parallel networks built from LSTM and transformer networks together with other neurons. Each receives every input word. These networks are highly connected to each other and to the second stage of the model. The first stage “programs” the model to create a “world” that has all the expected dependencies and relationships. The first stage can also increase the vocabulary of the model. The first stage may even set the appropriate number of networks and their connections in the second stage.

The response production/second stage is likely also massively parallel networks, and is likely connected perpendicularly to the reading comprehension/first stage. Notably both the reading comprehension and response production feedback regulation require the ability to understand every word available to use (including made up words and words in other languages) and put enough words together to understand complex concepts suggested in the text. This requires abstraction and linguistic logic.

Abstraction example: 
We generalize sounds, sights, tastes, etc. well. AI must as well.
E.g. a squirrel chitter is a high pitched sound. If something else makes that sound it also chitters.

Linguistic Logic: 
Humans often visualize things. We create an imaginary world in our minds eye in which events play out. AI must do something similar.
Eg, if you start a sentence with “running through the forest” it can’t continue “the dolphin jumped out of the water” unless it’s a kelp forest. It could also continue correctly, “he tripped over a root” or “she heard a parrot squawk” but not “the concert reached a crescendo”. 

The logical consistency/ factual accuracy depends on the context of the response. 
Eg for a question that requires a factual response, a brick cannot fly unless it is thrown, but in a fictional story it can do anything.

As far as the specific maths go, in the first stage of the model, the statistical operations appropriately alter the values of the short term/programmable memory matrices and the adjustable weights. In the second stage of the model, the matrices predict an appropriate next word.

The exact architecture now is a mystery because there’s still money to be made. Is the output fed back into both the first and second stages or just one of them? Do the model layers reflect the sentence structure e.g. does a past tense conjugation always goes to the same subnetwork? You may elucidate the architecture by creating a simple world with a limited number of words: a world of two people who live in different homes and go to the same grocery store to buy food (one item), work on the weekdays at the same place and on weekends play together in the park. Experiment with LSTM and transformer architectures and train the model to give simple responses to simple queries. Then train the model to build sentences with different ratings in the following categories: declarative/informational, descriptive, opinion (persuasive and other kinds), poetic/rhythmic, etc. At some point the training goes from human to AI. An AI model must be developed that can ask questions and evaluate answers (not create answers). There are many training cycles to determine which parameters to vary. The modifications are chosen by a sort of search algorithm (divide and conquer).

Why are the flow of the response and style variations coded in the weights of the model? There are an unlimited number of ways to construct a response. Style parameters: length of response, writing style/voice, sentence structure (noun or noun equivalent first in most, verb first in some commands, etc). You are not going to have a unique subnetwork for every kind of sentence. Variation is accomplished by changing the weights. These weights are determined by the input prompt and passed through the model. The difference between a weight and a regular matrix value is a weight is a coefficient used in the mathematical operations of a neuron, and a matrix value is a variable in those operations. 
Long responses are accomplished by breaking the writing down into parts. Each sentence has a symbolic purpose: description, action, fact, etc. Together they create a paragraph which itself has a symbolic purpose. Paragraphs are just related sentences and stories are composed of related paragraphs. Building complex related stories is based on combining related symbols. This reflects actual writing: “The last sentence said — so —. The last paragraph said — so —. The whole story up to here says — and I want to say — so —. Stylistically, I want suspense before the climax so I will start with — then do —”. 

How would you build something connected to an external database (like for a search engine):
Say you wanted to access something outside the model’s memory units. The first and second stage pause what they are doing and a “query unit” asks a database for information. This “query unit” would then return the information to both stage. What’s interesting is that this “query unit” resembles an entire LLM. It has to figure out what knowledge is missing from the memory units, form an appropriate question, read the search results and formulate an input into the active program.
Eg if the user says “Tell me about a lkajsdfl - rare cat breed discussed only in some obscure reddit thread,” it will search the internet for information about this cat and convey this information back to the matrices and weights.

Questions to test AI:
Ask about symbolism of an entirely new story.
Ask it to identify opinion vs fact in a social situation.
Ask it to draw connections between two stories.
Ask it to write a story with an invented word that has a very specific unique complex definition.


