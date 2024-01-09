'''
    Machine Learning methods for regression task
'''
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class Regressor():
    def __init__(self, data, train_idx, test_idx, y) -> None:
        '''
            data: numpy array (cell * gene)
            y: numpy array (cell * states)
        '''
        self.data = data
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.y = y
        self.X_train = self.data[self.train_idx, :]
        self.X_test = self.data[self.test_idx, :]
        self.y_train = self.y[self.train_idx, :]
        self.y_test = self.y[self.test_idx, :]
    
    def linear_regression(self):
        linear_reg = LinearRegression()        
        linear_reg.fit(self.X_train, self.y_train)        
        y_pred = linear_reg.predict(self.X_test)        
        mse = mean_squared_error(y_pred, self.y_test)
        mae = mean_absolute_error(y_pred, self.y_test)
        r2 = r2_score(y_pred, self.y_test)
        print("linear regression mse: {:.3f}, mae: {:.3f}, r2: {:.3f}".format(mse, mae, r2))
        return (mse, mae, r2)
    
    def random_forest(self):
        random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=32)
        # 训练模型
        random_forest_reg.fit(self.X_train, self.y_train)
        y_pred = random_forest_reg.predict(self.X_test)
        mse = mean_squared_error(y_pred, self.y_test)
        mae = mean_absolute_error(y_pred, self.y_test)
        r2 = r2_score(y_pred, self.y_test)
        print("random forest regression mse: {:.3f}, mae: {:.3f}, r2: {:.3f}".format(mse, mae, r2))
        return (mse, mae, r2)
        

    def lasso(self):        
        lasso_reg = Lasso(alpha=0.01)        
        lasso_reg.fit(self.X_train, self.y_train)        
        y_pred = lasso_reg.predict(self.X_test)
        mse = mean_squared_error(y_pred, self.y_test)
        mae = mean_absolute_error(y_pred, self.y_test)
        r2 = r2_score(y_pred, self.y_test)
        print("lasso mse: {:.3f}, mae: {:.3f}, r2: {:.3f}".format(mse, mae, r2))
        return (mse, mae, r2)
    
    def svr(self):
        svr_model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))  # 调整 C 和 epsilon 参数
        svr_model.fit(self.X_train, self.y_train)
        y_pred = svr_model.predict(self.X_test)
        mse = mean_squared_error(y_pred, self.y_test)
        mae = mean_absolute_error(y_pred, self.y_test)
        r2 = r2_score(y_pred, self.y_test)
        print("svr mse: {:.3f}, mae: {:.3f}, r2: {:.3f}".format(mse, mae, r2))
        return (mse, mae, r2)
        
