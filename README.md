# Question-Answering-Rnet
Implemented R-Net architecture with less number of GRUs. <br />
R-Net architecture consist three layers of GRUs to learn representation of questions and context. After that question-context gated attention architecture is used and than self attention architecture is used
to generate representation.<br />
Task is to identify starting and ending position in the given context. Output layer consist of GRU layer to predict starting and ending position.<br />
Categorical cross entropy loss function is used for both starting and ending position.<br />
Experimented with modification in self attention architecture and shown performance improvement on 25% of dataset with 29 Exact Match and 44.64 F1 score.
