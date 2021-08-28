import numpy as np 

def compute_error_for_given_points(b, m , data_points):
    totalError = 0
    for i in range(0, len(data_points)):
        x = data_points[i, 0]
        y = data_points[i, 1]

        #By the Error Formula
        totalError = totalError + (y - (m * x + b)) ** 2
    return totalError / float(len(data_points))

def setp_gradient(b_current,m_current,data_points,learningRt):
    b_gradient = 0
    m_gradient = 0
    N = float(len(data_points))
    for i in range(len(data_points)):
        x = data_points[i, 0]
        y = data_points[i, 1]
        b_gradient = b_gradient + -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient = m_gradient + -(2/N) * x * (y - ((m_current * x) + b_current))

    #Get the good convergence
    new_b = b_current - (learningRt * b_gradient)
    new_m = m_current - (learningRt * m_gradient)
    return [new_b, new_m]

def gradient_decent_runner(data_points, starting_m, starting_b, learning_rate, num_iter):
    b = starting_b
    m = starting_m

    for x in range(num_iter):
        b,m = setp_gradient(b,m, np.array(data_points), learning_rate)
    return [b,m]

def main():
    data_points = np.genfromtxt('data.csv', delimiter=',')

    #Hyperparameter (learning Rate)
    learning_rate = 0.0001
    num_iter = 1000

    # Slope Formula (y = mx+c)
    initial_m = 0
    initial_b = 0

    #Get the b,m optimal value
    [b,m] = gradient_decent_runner(data_points, initial_m, initial_b, learning_rate, num_iter)
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_given_points(initial_b, initial_m, data_points)))
    print("Running...")
    [b, m] = gradient_decent_runner(data_points, initial_m, initial_b, learning_rate, num_iter)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iter, b, m, compute_error_for_given_points(b, m, data_points)))






if __name__ == "__main__":
    main()