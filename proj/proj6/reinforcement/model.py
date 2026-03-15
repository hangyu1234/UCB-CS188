import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.numTrainingGames = 20000   # > 1000
        self.batch_size = 1
        hidden_size = 512
        self.W1 = nn.Parameter(state_dim, hidden_size)      # 第一层权重
        self.b1 = nn.Parameter(1, hidden_size)              # 第一层偏置
        self.W3 = nn.Parameter(hidden_size, action_dim)     # 输出层权重
        self.b3 = nn.Parameter(1, action_dim)               # 输出层偏置

        # 按前向传播顺序收集所有参数
        self.parameters = [self.W1, self.b1, self.W3, self.b3]


    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        Q_pred = self.run(states)                
        target = Q_target         
        loss = nn.SquareLoss(Q_pred, target)    
        return loss

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        x = states
        # 第一层：线性变换 + 偏置 + ReLU
        h1 = nn.Linear(x, self.W1)
        h1 = nn.AddBias(h1, self.b1)
        h1 = nn.ReLU(h1)


        # 输出层：线性变换 + 偏置（无激活）
        out = nn.Linear(h1, self.W3)
        out = nn.AddBias(out, self.b3)   
        return out

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states, Q_target)
        grads = nn.gradients(loss, self.parameters)
        for param, grad in zip(self.parameters, grads):
            param.update(grad, -self.learning_rate)