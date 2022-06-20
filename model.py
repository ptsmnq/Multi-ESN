import numpy as np


class EchoStateNetwork():


    def __init__(self, units=300, SR=0.99, input_shape=1, input_dims=1, output_shape=1, W_in_scale=0.1, W_res_scale=0.1, W_res_density=0.1, W_fb_scale=0.1, leak_rate=1.0, alpha=1.0e-4, seed=0, feedback=True):

        def W_res_create(shape, SR=0.99, scale=1.0, density=0.1):

            np.random.seed(self.seed)
            W1 = np.random.uniform(size=shape, low=-scale, high=scale)
            W2 = np.random.uniform(size=shape, low=0.0, high=1.0)
            W2 = (W2 > (1.0 - density)).astype(np.float)
            W = W1 * W2
            value, vector = np.linalg.eigh(W)
            sr = max(np.abs(value))
            return W * SR / sr

        self.units = units
        self.SR = SR
        self.input_shape = input_shape
        self.input_dims = input_dims
        self.output_shape = output_shape
        self.W_in_scale = W_in_scale
        self.W_res_scale = W_res_scale
        self.W_res_density = W_res_density
        self.W_fb_scale = W_fb_scale
        self.leak_rate = leak_rate
        self.alpha = alpha
        self.seed = seed
        self.feedback = feedback

        np.random.seed(seed=self.seed)
        self.W_in = np.random.uniform(size=(self.input_shape, self.units), low=-self.W_in_scale, high=self.W_in_scale)
        self.W_res = W_res_create(shape=(self.units, self.units), SR=self.SR, scale=self.W_res_scale, density=self.W_res_density)
        self.W_fb = np.random.uniform(size=(self.output_shape, self.units), low=-self.W_fb_scale, high=self.W_fb_scale)
        self.x_n = np.random.uniform(size=(self.input_dims, self.units))
        self.y_n_1 = np.random.uniform(size=(self.input_dims, self.output_shape))
        self.W_out =np.random.uniform(size=(self.units, self.output_shape))


    def fit(self, x, y):

        Xt_X, Xt_Y = 0.0, 0.0
        for i in range(x.shape[0]):
            In = np.matmul([x[i,:,:]], self.W_in)
            Res = np.matmul(self.x_n, self.W_res)
            Fb = np.matmul(self.y_n_1, self.W_fb)
            if self.feedback == True : self.x_n = np.tanh(In + Res + Fb).reshape(self.input_dims, self.units)
            else : self.x_n = ((1.0 - self.leak_rate) * self.x_n + self.leak_rate * np.tanh(In + Res)).reshape(self.input_dims, self.units)
            y_n = (y[i,:]).reshape(self.input_dims, self.output_shape)
            Xt_X += np.matmul(np.transpose(self.x_n), self.x_n)
            Xt_Y += np.matmul(np.transpose(self.x_n), y_n)
            self.y_n_1 = y_n
        Xt_X_aI = Xt_X + self.alpha * np.eye(int(self.units))
        self.W_out = np.matmul(np.linalg.inv(Xt_X_aI), Xt_Y)
        self.opt_x_n = self.x_n


    def predict(self, x):

        ans = []
        for i in range(x.shape[0]):
            In = np.matmul(x[i,:,:], self.W_in)
            Res = np.matmul(self.x_n, self.W_res)
            Fb = np.matmul(self.y_n_1, self.W_fb)
            if self.feedback == True : self.x_n = np.tanh(In + Res + Fb).reshape(self.input_dims, self.units)
            else : self.x_n = ((1.0 - self.leak_rate) * self.x_n + self.leak_rate * np.tanh(In + Res)).reshape(self.input_dims, self.units)
            pred = np.matmul(self.x_n, self.W_out)
            self.y_n_1 = pred
            ans.append(pred)

        return np.reshape(ans, (-1, self.input_dims))


    def freerun(self, x, sparse=1, pred_range=100):

        ans = []
        for i in range(pred_range):
            data = []
            for j in range(self.input_shape) : data.append(x[i+sparse*j,:])
            data = np.reshape(data, (self.input_dims, self.input_shape))
            In = np.matmul(data, self.W_in)
            Res = np.matmul(self.x_n, self.W_res)
            Fb = np.matmul(self.y_n_1, self.W_fb)
            if self.feedback == True : self.x_n = np.tanh(In + Res + Fb).reshape(self.input_dims, self.units)
            else : self.x_n = ((1.0 - self.leak_rate) * self.x_n + self.leak_rate * np.tanh(In + Res)).reshape(self.input_dims, self.units)
            pred = np.matmul(self.x_n, self.W_out)
            self.y_n_1 = pred
            ans.append(pred)
            x = np.append(x, pred).reshape(-1, self.input_dims)
        
        return np.reshape(ans, (-1, self.input_dims))
    

    def reset_reservoir(self):
        
        self.x_n = self.opt_x_n



class functions():


    def create_dataset(self, data, look_back=30, sparse=15, sample=3):

        self.data = data
        self.look_back = int(look_back)
        self.sparse = int(sparse)
        self.sample = int(sample)

        dataX, dataY = [], []
        for i in range(len(self.data)-self.look_back):
            x = []
            for j in range(self.sample) : x.append(self.data[i+self.sparse*j,:])
            dataX.append(x)
            dataY.append(self.data[i+look_back, :])

        return np.array(dataX).reshape(len(dataX), len(dataX[0][0]), len(dataX[0])), np.array(dataY)


    def rmse(self, real, pred):

        self.real = np.array(real).reshape(-1)
        self.pred = np.array(pred).reshape(-1)

        return np.sqrt(sum((self.real - self.pred) ** 2) / self.real.shape[0])
