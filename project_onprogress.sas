libname stat502 'm:\stat502';

*Load the training dataset;
filename trainimg 'm:\stat502\train-images.idx3-ubyte';
filename trainlab 'm:\stat502\train-labels.idx1-ubyte';
data train_img;
	infile trainimg recfm=n;
	input var PIB1.;
	array vars var2-var785;
	retain vars;
	if _n_>16 then do;
		index=mod(_n_-17,784)+1;
		vars[index]=var;
		if index=784 then output;
	end;
	drop var index;
run;

data train_label;
	infile trainlab recfm=n;
	input var1 PIB1.;
	if _n_>8;
run;

data mnist_train;
	merge train_label train_img;
run;

*Define an image classifier;
proc iml;
*Read train dataset and define them as matrix;
use mnist_train;
	read all var _all_ into training;
close mnist_train;

*x_train is for the training which has a matrix form with 42000*784;
x_train=training[,2:785];
*y_train contains a true label of data;
*y_temp contains a true label of data;
y_temp=training[,1];


show names;
*Make y_train form as [0,0,1,0,0,0,0,0,0,0] if 2 is true answer;
y_train=j(60000,10,0);
do i=1 to nrow(y_train);
	indx=y_temp[i]+1;
	y_train[i,indx]=1;
end;
*Check training data set;
t=y_train[1:100,];
print t;
* One shot hot vector output;

*Input data would be 100x784 -> 100 observations & 784 vectors
Weight 1 is 784x100 so their product would be 100x784 x 784x100= 100x100

*Initialize weights and biases here, we will use 3 layers with 1 hidden layer;
w1=randfun({784,100}, "Normal", 0,1);
b1=randfun({100,1}, "Normal",0,1);
w2=randfun({100,10}, "Normal", 0,1);
b2=randfun({10,1}, "Normal",0,1);
/* Define a sigmoid function */

* I defined ~ because it has overflow probelm ;
start lelu(s);
	r=nrow(s);
	c=ncol(s);
	zero=J(r,c,0);
	res=J(r,c,0)<>s;
	return res;
finish;

*How do we calculate softmax;
start softmax(x);
	x_ex=j(100,10,0);
	do i=1 to nrow(x);
		do j=1 to 10;
			c=max(x[i,]);
			x_ex[i,j]=exp(x[i,j]-c)/sum(exp(x[i,1:10]-c));
		end;
	end;
	return x_ex;
finish;	

start predict(x,w1,b1,w2,b2);
	b1rep=repeat(b1`,1,100);
	b1=shape(b1rep,100,100);
	a1=x*w1+b1; * (100*784)*(784*100)+(100*100);
	z1=lelu(a1);
	b2rep=repeat(b2`,1,100);
	b2s=shape(b2rep,100,10);
	a2=z1*w2+b2s; 
	y=softmax(a2);
	return y;
finish;

x_test=x_train[1:100,];
test1=predict(x_test, w1,b1,w2,b2);
print test1;
* This test works///////////////////////////////// ;



*Define loss function;
start cross_entropy_error(y,t);
	batch_size=nrow(y);
	h=-sum(t`*log(y+1e-7))/batch_size;
	return h;
finish;

start loss(pred_x,t);
	y=pred_x;
	loss=cross_entropy_error(y,t);
	return loss;
finish;

start accuracy(x,t,w1,b1,w2,b2);
	y=predict(x,w1,b1,w2,b2);
	y=loc(y=max(y));
	t=loc(t=max(t));
	if y=t then do;
		accuracy=accuracy+1;
	end;
	return accuracy;
finish;

start numerical_gradient(p,x1,t,w1,b1,w2,b2);
	h=1e-4;
	r_p=nrow(p);
	c_p=ncol(p);
	grad=j(r_p,c_p,0);
	do i=1 to r_p;
		do j=1 to c_p;
			tmp_val=x[i,j];
			x[i,j]=tmp_val+h;
			fxh1=loss(x1,t,w1,b1,w2,b2);

			x[i,j]=tmp_val-h;
			fxh2=loss(x1,t,w1,b1,w2,b2);
			grad[i,j]=(fxh1-fxh2)/(2*h);
			x[i,j]=tmp_val;
		end;
	end;
	return grad;
finish;


start gradient_descent(init_x);
	x=init_x;
	lr=0.1;
	step_num=1000;
	do i=1 to step_num;
		grad=numerical_gradient(x,t,w1,b1,w2,b2);
		x=x-lr*grad;
	end;
	return x;
finish;


*Test gradient method by using batch dataset;
rand_batch=sample(1:nrow(x_train), 100);	*Obtain 100 random samples;
x_batch=x_train[rand_batch,];
x_lab_batch=y_train[rand_batch,];

print x_lab_batch x_batch;
show names;
	gradw1=numerical_gradient(w1,x_test,t_batch,w1,b1,w2,b2);
	w1=w1-learning_rate*gradw1; *Update w1;
end;

	p=loc(y=max(y));

end;


	rand_batch=sample(1:nrow(x_c),10);
	x_batch=x_c[rand_batch,];
	y_batch=predict(x_batch,w1,b1,w2,b2);
end;

grad=numerical_gradient(x_batch,t_batch,w1,b1,w2,b2);

	do j=1 to ncol(x_batch);
		grad=grad-learning_rate*grad[i,j];
	end;
	loss=loss(x_batch,t_batch,w1,b1,w2,b2);
end;

print x_batch;

print t_batch;
* Show defined variables so far;
show names;




/*


h=-sum(t`*log(y+1e-7));
print h;
t={0,0,1,0,0,0,0,0,0,0};

y={0.1,0.05,0.1, 0.0,0.05,0.1,0.0,0.6,0,0};
print y;
k=cross_entropy_error(y,t);
print k;


t={1,0,0,0,0};
do i=1 to nrow(x_train);
	x_b=x_train[i,];
	y=predict(x_b,w1,b1,w2,b2);
	p=loc(y=max(y));
	if p=t[i,] then do;
		accuracy_cnt=accuracy_cnt+1;
	end;
	
	put 'accuracy:'+accuracy_cny/nrow(x_train);
end;



start mean_squared_error(y,t);
	return 0.5*sum((y-t)##2);
finish;

