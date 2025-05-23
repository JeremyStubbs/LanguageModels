**Large Language Models**

**Evolution**
They evolved from translation models. The structure is an encoder-decoder which is explained in the foundational papers for language models. Kalchbrenner and Blunson kind of had an encoder-decoder model in 2013, but it was made explicit by Bengio's lab in 2014. Bengio also introduced attention in 2014.

Kalchbrenner and Blunson's encoder passed one matrix to decoder. Bengio in 2014 figured you would get better results if you passed the matrices from all the intermediate steps leading to the final encoding matrix to the decoder and did a weighted sum of them. Jakob uzkoreit figured that the concept of generating extra information and doing a weighted sum (in this case applied to just the input) could be used to make the encoder parallel instead of sequential when you have positional encoding - this is called self-attention. The other transformer authors, mainly shazeer, then found a clever sinuisoidal positional encoding, a more clever way to do a weighted sum with Query-Key-Value (almost a search algorithm), applied self-attention to the decoder, and used multiple lines of different weighted sums simultaneously - called multi-headed attention. 

**Conceptual Understading**

LLMs do not operate on Boolean logic, mathematics on 0s and 1s (although the transistor does), but rather statistics - groups of neurons perform complex statistical operations on matrices. Chatgpt told me this and Geoffrey Hinton is famous partly for modeling networks as Boltzman fields (stats stuff). The statistics determine 1) most likely meaning of a word or phrase and 2) how to craft a response.

It is helpful to understand how the human mind works. If I tell you a story, you will (assuming average intelligence) be able to draw conclusions about events in it and make predictions about it. Much of this revolves around creating a visualization of the story. Your mind has much practice in visualizing things that happen and assigning all the appropriate physics to it (time, gravity, size and speed of things). Your brain works on rather simple neurons - threshold activation. It accomplishes this task by using extremely complex networks. I imagine the brain as a humming mass of web-like electricity. You cannot make these sorts of 3D connections on silicon, but you don’t need to because silicon neurons can do extremely complex matrix/statistical operations. But even with silicon there will be some similarity at least functionally to accomplish these tasks.

The structures that make this possible are programmable memory units called long-short term memory networks and transformer networks. LSTM and transformers to store matrix values between time steps of the program. These matrices are analogous to the context vectors of early models, but now they must encompass a lot more information: content comprehension and style directives of the response (length, writing style, etc). In summary, words are passed in one by one and used to create matrices that encompass all their relationships.

I believe that the architecture of the first stage of the model is massively parallel input networks. There are many parallel networks built from LSTM and transformer networks together with other neurons. These networks are highly connected to each other and to the second stage of the model. Each receives every input word. The first stage “programs” both stages of the model to create a “world” that has all the expected dependencies and relationships.

The response production/second stage is likely also massively parallel networks and is likely connected perpendicularly to the reading comprehension/first stage.

As I mentioned, the first task is reading comprehension done by the first stage. The first stage makes sense of the prompt and provides directives to the second stage. Most likely all the words are processed by the first stage before the response production starts by the second stage. The first stage then passes matrix(ces) and weights to the second stage which tell it how to begin the response. The first stage sets the appropriate number of networks and their connections in both stages. The first stage can change default settings and increase the vocabulary of the model.

The response production/second stage makes sentences one word at a time. I don’t think it’s physically possible to do it any other way. It ensures flow and completion through feedforward and feedback mechanisms with both itself and the first stage after each word. There is constant evaluation of what the second stage is doing and readjustments of the matrix values, weights, and/or network connections of the second stage to accomplish its directives.

As far as the specific maths go, the statistical operations appropriately alter the values of the short term/programmable memory matrices and the adjustable weights. In addition, it selects the number of networks to use and their connections. In the second stage of the model, the matrices predict an appropriate next word.

Both the reading comprehension and response production feedback regulation require abstraction and linguistic logic - the ability to understand every word available to use (including made up words) and put enough words together to understand complex concepts. **Said another way,** both to go from words to matrices and go from matrices to words you have to be able to create complex relationships between tensor representations of words. There must be some default settings that store the physics and definitions: values in the matrices and intra and inter connections in the networks. But these default values must be easily accessible because they can be changed. New relationships can be also be given in the prompt.

Even a simple phrase like Jack and Jill are dating implies a whole lotta information. Both stages set matrix values and create new relationships within and between networks to figure stuff out with linguistic logic, and express unique sentences dependent on those relationships. Depending on the environment, different physics have to be implemented. By changing default matrix values?

You can make up words and you can redefine words, so you must be able to easily access and change matrix values and intra and inter network relationships. For example, I can say “make a sentence with the word circular, but circular means flegblarg which is the state of being too sick to get drunk enough to pick up a slut at a bar.” And ChatGPT will spit out a perfect sentence.

Abstraction example:  
We generalize sounds, sights, tastes, etc. well. AI must as well.

E.g. a squirrel chitter is a high pitched sound. If something else makes that sound it also chitters.

Linguistic Logic:  
Humans often visualize things. We create an imaginary world in our minds eye in which events play out. AI must do something similar.

Eg, if you start a sentence with “running through the forest” it can’t continue “the dolphin jumped out of the water” unless it’s a kelp forest. It could also continue correctly, “he tripped over a root” or “she heard a parrot squawk” but not “the concert reached a crescendo”.

The logical consistency/ factual accuracy depends on the context of the response.

Eg for a question that requires a factual response, a brick cannot fly unless it is thrown, but in a fictional story it can do anything.

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

