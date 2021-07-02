f <- function(net) {
	return ( 1 / (1 + exp(-net)))
}

df_net <- function(f_net) {
	return (f_net * (1 - f_net ))
}

mlp.architecture <- function(input.length = 2,
							 	hidden.length = 2,
								output.length = 1,
								activation.function = f,
								d_activation.function = df_net ){
	model = list()
	model$input.length = input.length
	model$hidden.length = hidden.length 
	model$output.length = output.length 

	#   2 neoronios 
	#	1	w_11	w_12	theta_1
	#	2   w_21	w_22	theta_2

	# inicializando camada escondida inicializada
	# com numeros aleatorios com base nos tamanhos passados

	# inicializando camada escondida inicializada
	# com numeros aleatorios com base nos tamanhos passados
	#
	# hidden = tamanho da camada escondida * tamanho camada de entrada + 1 (representa theta)
	# nrow = tamanho camada de escondida 
	# ncol = tamanho camada de entrada
	
	model$hidden = matrix(runif(min=-0.5, max=0.5,
					hidden.length * (input.length+1)),
					nrow=hidden.length, ncol=input.length+1)

	# inicializando camada de saida 
	model$output = matrix(runif(min=-0.5, max=0.5,
					output.length * (hidden.length+1)),
					nrow=output.length, ncol=hidden.length+1)

	model$f = activation.function
	model$df_dnet = d_activation.function 
	return (model)
	
}
# XOR
# 0 0 0
# 0 1 1
# 1 0 1
# 1 1 0

# Xp = c(0,1)

mlp.forward <- function(model, Xp){
	# Hidden layer
	# multiplica todas entradas pelos pesos
	# resultando no net_h_p
	# na sequencia aplica a funcao de ativacao
	net_h_p = model$hidden %*% c(Xp,1)
	f_net_h_p = model$f(net_h_p)

	# Output layer 
	net_o_p = model$output %*% c(as.numeric(f_net_h_p), 1)
	f_net_o_p = model$f(net_o_p)

	# Results 

	ret = list()
	ret$net_h_p = net_h_p
	ret$net_o_p = net_o_p
	ret$f_net_h_p = f_net_h_p
	ret$f_net_o_p = f_net_o_p

	return (ret)

}

mlp.backpropagation <- function(model, dataset, eta=0.1, threshold =1e-3){
	squaredError = 2 * threshold
	counter = 0 

	while (squaredError > threshold ){
		squaredError = 0
		for (p in 1:nrow(dataset)){
			Xp = as.numeric(dataset[p, 1:model$input.length])
			Yp = as.numeric(dataset[p,
					(model$input.length+1):ncol(dataset)])
	
			results = mlp.forward(model, Xp)
			Op = results$f_net_o_p

			# Calculando o erro 
			error = Yp - Op

			squaredError = squaredError + sum(error^2)

			# Trainig output 
			# delta_o = (Yp - Op) * f_o_p'(net_o_p)
			#
			# w(t+1) = w(t) - eta * dE2_dw
			# w(t+1) = w(t) - eta * delta_o * i_pj

			delta_o_p = error * model$df_dnet(results$f_net_o_p)

			# Training hidden 
			#
			# delta_h = f_h_p'(net_h_p) * sum(delta_o * w_o_kj)
			# w(t+1) = w(t) - eta * delta_h * Xp

			w_o_kj = model$output[,1:model$hidden.length]
			delta_h_p = 
				as.numeric(model$df_dnet(results$f_net_h_p)) *
					(as.numeric(delta_o_p) %*% w_o_kj) 

			# Training 
			model$output = model$output + 
				eta * ( delta_o_p%*% as.vector(c(results$f_net_h_p, 1)))
			model$hidden = model$hidden +
				eta * (t(delta_h_p)%*%as.vector(c(Xp,1)))

			squaredError = squaredError / nrow(dataset)

			cat("Erro medio quadrado = ", squaredError, "\n")

			counter = counter + 1
		}
	}

	ret = list()
	ret$model = model 
	ret$counter = counter 

	return (ret)
}
