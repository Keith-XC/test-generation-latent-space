ю╙
Щ¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02unknown8╞╔	
d
VariableVarHandleOp*
_output_shapes
: *
dtype0
*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0

В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
Ж
conv2d_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_1_1/kernel

%conv2d_1_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_1/kernel*&
_output_shapes
: @*
dtype0
v
conv2d_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_1_1/bias
o
#conv2d_1_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1_1/bias*
_output_shapes
:@*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АHА*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
АHА*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:А*
dtype0
}
dense_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*!
shared_namedense_1_1/kernel
v
$dense_1_1/kernel/Read/ReadVariableOpReadVariableOpdense_1_1/kernel*
_output_shapes
:	А
*
dtype0
t
dense_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1_1/bias
m
"dense_1_1/bias/Read/ReadVariableOpReadVariableOpdense_1_1/bias*
_output_shapes
:
*
dtype0
n
Adadelta/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdadelta/iter
g
!Adadelta/iter/Read/ReadVariableOpReadVariableOpAdadelta/iter*
_output_shapes
: *
dtype0	
p
Adadelta/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/decay
i
"Adadelta/decay/Read/ReadVariableOpReadVariableOpAdadelta/decay*
_output_shapes
: *
dtype0
А
Adadelta/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdadelta/learning_rate
y
*Adadelta/learning_rate/Read/ReadVariableOpReadVariableOpAdadelta/learning_rate*
_output_shapes
: *
dtype0
l
Adadelta/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/rho
e
 Adadelta/rho/Read/ReadVariableOpReadVariableOpAdadelta/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
к
#Adadelta/conv2d_2/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adadelta/conv2d_2/kernel/accum_grad
г
7Adadelta/conv2d_2/kernel/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_2/kernel/accum_grad*&
_output_shapes
: *
dtype0
Ъ
!Adadelta/conv2d_2/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adadelta/conv2d_2/bias/accum_grad
У
5Adadelta/conv2d_2/bias/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/conv2d_2/bias/accum_grad*
_output_shapes
: *
dtype0
о
%Adadelta/conv2d_1_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%Adadelta/conv2d_1_1/kernel/accum_grad
з
9Adadelta/conv2d_1_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/conv2d_1_1/kernel/accum_grad*&
_output_shapes
: @*
dtype0
Ю
#Adadelta/conv2d_1_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adadelta/conv2d_1_1/bias/accum_grad
Ч
7Adadelta/conv2d_1_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/conv2d_1_1/bias/accum_grad*
_output_shapes
:@*
dtype0
в
"Adadelta/dense_2/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АHА*3
shared_name$"Adadelta/dense_2/kernel/accum_grad
Ы
6Adadelta/dense_2/kernel/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_2/kernel/accum_grad* 
_output_shapes
:
АHА*
dtype0
Щ
 Adadelta/dense_2/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adadelta/dense_2/bias/accum_grad
Т
4Adadelta/dense_2/bias/accum_grad/Read/ReadVariableOpReadVariableOp Adadelta/dense_2/bias/accum_grad*
_output_shapes	
:А*
dtype0
е
$Adadelta/dense_1_1/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*5
shared_name&$Adadelta/dense_1_1/kernel/accum_grad
Ю
8Adadelta/dense_1_1/kernel/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/dense_1_1/kernel/accum_grad*
_output_shapes
:	А
*
dtype0
Ь
"Adadelta/dense_1_1/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"Adadelta/dense_1_1/bias/accum_grad
Х
6Adadelta/dense_1_1/bias/accum_grad/Read/ReadVariableOpReadVariableOp"Adadelta/dense_1_1/bias/accum_grad*
_output_shapes
:
*
dtype0
и
"Adadelta/conv2d_2/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adadelta/conv2d_2/kernel/accum_var
б
6Adadelta/conv2d_2/kernel/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_2/kernel/accum_var*&
_output_shapes
: *
dtype0
Ш
 Adadelta/conv2d_2/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adadelta/conv2d_2/bias/accum_var
С
4Adadelta/conv2d_2/bias/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/conv2d_2/bias/accum_var*
_output_shapes
: *
dtype0
м
$Adadelta/conv2d_1_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*5
shared_name&$Adadelta/conv2d_1_1/kernel/accum_var
е
8Adadelta/conv2d_1_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/conv2d_1_1/kernel/accum_var*&
_output_shapes
: @*
dtype0
Ь
"Adadelta/conv2d_1_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adadelta/conv2d_1_1/bias/accum_var
Х
6Adadelta/conv2d_1_1/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/conv2d_1_1/bias/accum_var*
_output_shapes
:@*
dtype0
а
!Adadelta/dense_2/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АHА*2
shared_name#!Adadelta/dense_2/kernel/accum_var
Щ
5Adadelta/dense_2/kernel/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_2/kernel/accum_var* 
_output_shapes
:
АHА*
dtype0
Ч
Adadelta/dense_2/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*0
shared_name!Adadelta/dense_2/bias/accum_var
Р
3Adadelta/dense_2/bias/accum_var/Read/ReadVariableOpReadVariableOpAdadelta/dense_2/bias/accum_var*
_output_shapes	
:А*
dtype0
г
#Adadelta/dense_1_1/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*4
shared_name%#Adadelta/dense_1_1/kernel/accum_var
Ь
7Adadelta/dense_1_1/kernel/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/dense_1_1/kernel/accum_var*
_output_shapes
:	А
*
dtype0
Ъ
!Adadelta/dense_1_1/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!Adadelta/dense_1_1/bias/accum_var
У
5Adadelta/dense_1_1/bias/accum_var/Read/ReadVariableOpReadVariableOp!Adadelta/dense_1_1/bias/accum_var*
_output_shapes
:
*
dtype0

NoOpNoOp
ы;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ж;
valueЬ;BЩ; BТ;
·
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	stochastic_mode_tensor
	_stochastic_mode_tensor

	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
R
.	variables
/regularization_losses
0trainable_variables
1	keras_api
h

2kernel
3bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
OM
VARIABLE_VALUEVariable1stochastic_mode_tensor/.ATTRIBUTES/VARIABLE_VALUE
═
8iter
	9decay
:learning_rate
;rho
accum_gradt
accum_gradu
accum_gradv
accum_gradw(
accum_gradx)
accum_grady2
accum_gradz3
accum_grad{	accum_var|	accum_var}	accum_var~	accum_var(	accum_varА)	accum_varБ2	accum_varВ3	accum_varГ
?
0
1
2
3
(4
)5
26
37
	8
 
8
0
1
2
3
(4
)5
26
37
н
<layer_metrics

=layers
>metrics
?non_trainable_variables
	variables
regularization_losses
trainable_variables
@layer_regularization_losses
 
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
Alayer_metrics

Blayers
Cmetrics
Dnon_trainable_variables
	variables
regularization_losses
trainable_variables
Elayer_regularization_losses
][
VARIABLE_VALUEconv2d_1_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_1_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
Flayer_metrics

Glayers
Hmetrics
Inon_trainable_variables
	variables
regularization_losses
trainable_variables
Jlayer_regularization_losses
 
 
 
н
Klayer_metrics

Llayers
Mmetrics
Nnon_trainable_variables
	variables
regularization_losses
trainable_variables
Olayer_regularization_losses
 
 
 
н
Player_metrics

Qlayers
Rmetrics
Snon_trainable_variables
 	variables
!regularization_losses
"trainable_variables
Tlayer_regularization_losses
 
 
 
н
Ulayer_metrics

Vlayers
Wmetrics
Xnon_trainable_variables
$	variables
%regularization_losses
&trainable_variables
Ylayer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
н
Zlayer_metrics

[layers
\metrics
]non_trainable_variables
*	variables
+regularization_losses
,trainable_variables
^layer_regularization_losses
 
 
 
н
_layer_metrics

`layers
ametrics
bnon_trainable_variables
.	variables
/regularization_losses
0trainable_variables
clayer_regularization_losses
\Z
VARIABLE_VALUEdense_1_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_1_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31
 

20
31
н
dlayer_metrics

elayers
fmetrics
gnon_trainable_variables
4	variables
5regularization_losses
6trainable_variables
hlayer_regularization_losses
LJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

i0
j1

	0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ktotal
	lcount
m	variables
n	keras_api
D
	ototal
	pcount
q
_fn_kwargs
r	variables
s	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

m	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

o0
p1

r	variables
ХТ
VARIABLE_VALUE#Adadelta/conv2d_2/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE!Adadelta/conv2d_2/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE%Adadelta/conv2d_1_1/kernel/accum_grad[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE#Adadelta/conv2d_1_1/bias/accum_gradYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE"Adadelta/dense_2/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE Adadelta/dense_2/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE$Adadelta/dense_1_1/kernel/accum_grad[layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUE"Adadelta/dense_1_1/bias/accum_gradYlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE"Adadelta/conv2d_2/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE Adadelta/conv2d_2/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE$Adadelta/conv2d_1_1/kernel/accum_varZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE"Adadelta/conv2d_1_1/bias/accum_varXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUE!Adadelta/dense_2/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
ОЛ
VARIABLE_VALUEAdadelta/dense_2/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE#Adadelta/dense_1_1/kernel/accum_varZlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE!Adadelta/dense_1_1/bias/accum_varXlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
П
serving_default_conv2d_inputPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
╡
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d_2/kernelconv2d_2/biasconv2d_1_1/kernelconv2d_1_1/biasVariabledense_2/kerneldense_2/biasdense_1_1/kerneldense_1_1/bias*
Tin
2
*
Tout
2*'
_output_shapes
:         
*+
_read_only_resource_inputs
		**
config_proto

GPU 

CPU2J 8*-
f(R&
$__inference_signature_wrapper_377303
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
с
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp%conv2d_1_1/kernel/Read/ReadVariableOp#conv2d_1_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp$dense_1_1/kernel/Read/ReadVariableOp"dense_1_1/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adadelta/conv2d_2/kernel/accum_grad/Read/ReadVariableOp5Adadelta/conv2d_2/bias/accum_grad/Read/ReadVariableOp9Adadelta/conv2d_1_1/kernel/accum_grad/Read/ReadVariableOp7Adadelta/conv2d_1_1/bias/accum_grad/Read/ReadVariableOp6Adadelta/dense_2/kernel/accum_grad/Read/ReadVariableOp4Adadelta/dense_2/bias/accum_grad/Read/ReadVariableOp8Adadelta/dense_1_1/kernel/accum_grad/Read/ReadVariableOp6Adadelta/dense_1_1/bias/accum_grad/Read/ReadVariableOp6Adadelta/conv2d_2/kernel/accum_var/Read/ReadVariableOp4Adadelta/conv2d_2/bias/accum_var/Read/ReadVariableOp8Adadelta/conv2d_1_1/kernel/accum_var/Read/ReadVariableOp6Adadelta/conv2d_1_1/bias/accum_var/Read/ReadVariableOp5Adadelta/dense_2/kernel/accum_var/Read/ReadVariableOp3Adadelta/dense_2/bias/accum_var/Read/ReadVariableOp7Adadelta/dense_1_1/kernel/accum_var/Read/ReadVariableOp5Adadelta/dense_1_1/bias/accum_var/Read/ReadVariableOpConst*.
Tin'
%2#
	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__traced_save_377834
╚
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariableconv2d_2/kernelconv2d_2/biasconv2d_1_1/kernelconv2d_1_1/biasdense_2/kerneldense_2/biasdense_1_1/kerneldense_1_1/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcounttotal_1count_1#Adadelta/conv2d_2/kernel/accum_grad!Adadelta/conv2d_2/bias/accum_grad%Adadelta/conv2d_1_1/kernel/accum_grad#Adadelta/conv2d_1_1/bias/accum_grad"Adadelta/dense_2/kernel/accum_grad Adadelta/dense_2/bias/accum_grad$Adadelta/dense_1_1/kernel/accum_grad"Adadelta/dense_1_1/bias/accum_grad"Adadelta/conv2d_2/kernel/accum_var Adadelta/conv2d_2/bias/accum_var$Adadelta/conv2d_1_1/kernel/accum_var"Adadelta/conv2d_1_1/bias/accum_var!Adadelta/dense_2/kernel/accum_varAdadelta/dense_2/bias/accum_var#Adadelta/dense_1_1/kernel/accum_var!Adadelta/dense_1_1/bias/accum_var*-
Tin&
$2"*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__traced_restore_377945▒м
ы
й
A__inference_dense_layer_call_and_return_conditional_losses_377013

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         АH:::P L
(
_output_shapes
:         АH
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╗
_
C__inference_flatten_layer_call_and_return_conditional_losses_377592

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АH2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         АH2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
■

Y
*uwiz_bernoulli_dropout_1_cond_false_377359
dropout_mul_dense_relu
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstД
dropout/MulMuldropout_mul_dense_reludropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Muld
dropout/ShapeShapedropout_mul_dense_relu*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
╣
Ъ
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_376969

inputs 
cond_readvariableop_resource
identityИвcond
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
: *
dtype0
2
cond/ReadVariableOp╨
condIfcond/ReadVariableOp:value:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_376948*.
output_shapes
:         @*#
then_branchR
cond_true_3769472
conds
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:         @2
cond/Identityy
IdentityIdentitycond/Identity:output:0^cond*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         @:2
condcond:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: 
г

ш
$__inference_signature_wrapper_377303
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityИвStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*'
_output_shapes
:         
*+
_read_only_resource_inputs
		**
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__wrapped_model_3768412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Ї
{
&__inference_dense_layer_call_fn_377617

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3770132
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         АH::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         АH
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╒
a
'uwiz_bernoulli_dropout_cond_true_377321%
!dropout_mul_max_pooling2d_maxpool
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/ConstЦ
dropout/MulMul!dropout_mul_max_pooling2d_maxpooldropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mulo
dropout/ShapeShape!dropout_mul_max_pooling2d_maxpool*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
Х5
╝
F__inference_sequential_layer_call_and_return_conditional_losses_377469

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource7
3uwiz_bernoulli_dropout_cond_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИвuwiz_bernoulli_dropout/condвuwiz_bernoulli_dropout_1/condк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╣
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╥
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Relu├
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool─
*uwiz_bernoulli_dropout/cond/ReadVariableOpReadVariableOp3uwiz_bernoulli_dropout_cond_readvariableop_resource*
_output_shapes
: *
dtype0
2,
*uwiz_bernoulli_dropout/cond/ReadVariableOp█
uwiz_bernoulli_dropout/condIf2uwiz_bernoulli_dropout/cond/ReadVariableOp:value:0max_pooling2d/MaxPool:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *;
else_branch,R*
(uwiz_bernoulli_dropout_cond_false_377412*.
output_shapes
:         @*:
then_branch+R)
'uwiz_bernoulli_dropout_cond_true_3774112
uwiz_bernoulli_dropout/cond╕
$uwiz_bernoulli_dropout/cond/IdentityIdentity$uwiz_bernoulli_dropout/cond:output:0*
T0*/
_output_shapes
:         @2&
$uwiz_bernoulli_dropout/cond/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
flatten/Constз
flatten/ReshapeReshape-uwiz_bernoulli_dropout/cond/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         АH2
flatten/Reshapeб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2

dense/Relu╚
,uwiz_bernoulli_dropout_1/cond/ReadVariableOpReadVariableOp3uwiz_bernoulli_dropout_cond_readvariableop_resource*
_output_shapes
: *
dtype0
2.
,uwiz_bernoulli_dropout_1/cond/ReadVariableOpя
uwiz_bernoulli_dropout_1/condIf4uwiz_bernoulli_dropout_1/cond/ReadVariableOp:value:0dense/Relu:activations:0^uwiz_bernoulli_dropout/cond*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:         А* 
_read_only_resource_inputs
 *=
else_branch.R,
*uwiz_bernoulli_dropout_1_cond_false_377442*'
output_shapes
:         А*<
then_branch-R+
)uwiz_bernoulli_dropout_1_cond_true_3774412
uwiz_bernoulli_dropout_1/cond╖
&uwiz_bernoulli_dropout_1/cond/IdentityIdentity&uwiz_bernoulli_dropout_1/cond:output:0*
T0*(
_output_shapes
:         А2(
&uwiz_bernoulli_dropout_1/cond/Identityж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
dense_1/MatMul/ReadVariableOp┤
dense_1/MatMulMatMul/uwiz_bernoulli_dropout_1/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_1/Softmaxл
IdentityIdentitydense_1/Softmax:softmax:0^uwiz_bernoulli_dropout/cond^uwiz_bernoulli_dropout_1/cond*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::2:
uwiz_bernoulli_dropout/conduwiz_bernoulli_dropout/cond2>
uwiz_bernoulli_dropout_1/conduwiz_bernoulli_dropout_1/cond:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
С
Ь
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_377674

inputs 
cond_readvariableop_resource
identityИвcond
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
: *
dtype0
2
cond/ReadVariableOp┬
condIfcond/ReadVariableOp:value:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:         А* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_377653*'
output_shapes
:         А*#
then_branchR
cond_true_3776522
condl
cond/IdentityIdentitycond:output:0*
T0*(
_output_shapes
:         А2
cond/Identityr
IdentityIdentitycond/Identity:output:0^cond*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А:2
condcond:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: 
┘

<
cond_false_377621
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstА
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
╒
a
'uwiz_bernoulli_dropout_cond_true_377411%
!dropout_mul_max_pooling2d_maxpool
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/ConstЦ
dropout/MulMul!dropout_mul_max_pooling2d_maxpooldropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mulo
dropout/ShapeShape!dropout_mul_max_pooling2d_maxpool*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
С
Ь
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_377086

inputs 
cond_readvariableop_resource
identityИвcond
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
: *
dtype0
2
cond/ReadVariableOp┬
condIfcond/ReadVariableOp:value:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:         А* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_377065*'
output_shapes
:         А*#
then_branchR
cond_true_3770642
condl
cond/IdentityIdentitycond:output:0*
T0*(
_output_shapes
:         А2
cond/Identityr
IdentityIdentitycond/Identity:output:0^cond*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А:2
condcond:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: 
д?
Ц
!__inference__wrapped_model_376841
conv2d_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resourceB
>sequential_uwiz_bernoulli_dropout_cond_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identityИв&sequential/uwiz_bernoulli_dropout/condв(sequential/uwiz_bernoulli_dropout_1/cond╦
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpр
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
sequential/conv2d/Conv2D┬
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp╨
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
sequential/conv2d/BiasAddЦ
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential/conv2d/Relu╤
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp■
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D╚
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp╪
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
sequential/conv2d_1/BiasAddЬ
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
sequential/conv2d_1/Reluф
 sequential/max_pooling2d/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolх
5sequential/uwiz_bernoulli_dropout/cond/ReadVariableOpReadVariableOp>sequential_uwiz_bernoulli_dropout_cond_readvariableop_resource*
_output_shapes
: *
dtype0
27
5sequential/uwiz_bernoulli_dropout/cond/ReadVariableOpЭ
&sequential/uwiz_bernoulli_dropout/condIf=sequential/uwiz_bernoulli_dropout/cond/ReadVariableOp:value:0)sequential/max_pooling2d/MaxPool:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *F
else_branch7R5
3sequential_uwiz_bernoulli_dropout_cond_false_376784*.
output_shapes
:         @*E
then_branch6R4
2sequential_uwiz_bernoulli_dropout_cond_true_3767832(
&sequential/uwiz_bernoulli_dropout/cond┘
/sequential/uwiz_bernoulli_dropout/cond/IdentityIdentity/sequential/uwiz_bernoulli_dropout/cond:output:0*
T0*/
_output_shapes
:         @21
/sequential/uwiz_bernoulli_dropout/cond/IdentityЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
sequential/flatten/Const╙
sequential/flatten/ReshapeReshape8sequential/uwiz_bernoulli_dropout/cond/Identity:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:         АH2
sequential/flatten/Reshape┬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02(
&sequential/dense/MatMul/ReadVariableOp─
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential/dense/MatMul└
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp╞
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential/dense/BiasAddМ
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential/dense/Reluщ
7sequential/uwiz_bernoulli_dropout_1/cond/ReadVariableOpReadVariableOp>sequential_uwiz_bernoulli_dropout_cond_readvariableop_resource*
_output_shapes
: *
dtype0
29
7sequential/uwiz_bernoulli_dropout_1/cond/ReadVariableOp╝
(sequential/uwiz_bernoulli_dropout_1/condIf?sequential/uwiz_bernoulli_dropout_1/cond/ReadVariableOp:value:0#sequential/dense/Relu:activations:0'^sequential/uwiz_bernoulli_dropout/cond*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:         А* 
_read_only_resource_inputs
 *H
else_branch9R7
5sequential_uwiz_bernoulli_dropout_1_cond_false_376814*'
output_shapes
:         А*G
then_branch8R6
4sequential_uwiz_bernoulli_dropout_1_cond_true_3768132*
(sequential/uwiz_bernoulli_dropout_1/cond╪
1sequential/uwiz_bernoulli_dropout_1/cond/IdentityIdentity1sequential/uwiz_bernoulli_dropout_1/cond:output:0*
T0*(
_output_shapes
:         А23
1sequential/uwiz_bernoulli_dropout_1/cond/Identity╟
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpр
sequential/dense_1/MatMulMatMul:sequential/uwiz_bernoulli_dropout_1/cond/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential/dense_1/MatMul┼
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp═
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential/dense_1/BiasAddЪ
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
sequential/dense_1/Softmax╠
IdentityIdentity$sequential/dense_1/Softmax:softmax:0'^sequential/uwiz_bernoulli_dropout/cond)^sequential/uwiz_bernoulli_dropout_1/cond*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::2P
&sequential/uwiz_bernoulli_dropout/cond&sequential/uwiz_bernoulli_dropout/cond2T
(sequential/uwiz_bernoulli_dropout_1/cond(sequential/uwiz_bernoulli_dropout_1/cond:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
▒

к
B__inference_conv2d_layer_call_and_return_conditional_losses_376853

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
▐
~
)__inference_conv2d_1_layer_call_fn_376885

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3768752
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┼'
Ф
F__inference_sequential_layer_call_and_return_conditional_losses_377163
conv2d_input
conv2d_377135
conv2d_377137
conv2d_1_377140
conv2d_1_377142!
uwiz_bernoulli_dropout_377146
dense_377150
dense_377152
dense_1_377157
dense_1_377159
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв.uwiz_bernoulli_dropout/StatefulPartitionedCallв0uwiz_bernoulli_dropout_1/StatefulPartitionedCallЎ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_377135conv2d_377137*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3768532 
conv2d/StatefulPartitionedCallЫ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_377140conv2d_1_377142*
Tin
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3768752"
 conv2d_1/StatefulPartitionedCallь
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         @* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3768912
max_pooling2d/PartitionedCall┐
.uwiz_bernoulli_dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0uwiz_bernoulli_dropout_377146*
Tin
2*
Tout
2*/
_output_shapes
:         @*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*[
fVRT
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_37696920
.uwiz_bernoulli_dropout/StatefulPartitionedCallс
flatten/PartitionedCallPartitionedCall7uwiz_bernoulli_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         АH* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3769942
flatten/PartitionedCall■
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_377150dense_377152*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3770132
dense/StatefulPartitionedCall╛
0uwiz_bernoulli_dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0uwiz_bernoulli_dropout_377146*
Tin
2*
Tout
2*(
_output_shapes
:         А*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*]
fXRV
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_37708622
0uwiz_bernoulli_dropout_1/StatefulPartitionedCallа
dense_1/StatefulPartitionedCallStatefulPartitionedCall9uwiz_bernoulli_dropout_1/StatefulPartitionedCall:output:0dense_1_377157dense_1_377159*
Tin
2*
Tout
2*'
_output_shapes
:         
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3771152!
dense_1/StatefulPartitionedCallц
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall/^uwiz_bernoulli_dropout/StatefulPartitionedCall1^uwiz_bernoulli_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.uwiz_bernoulli_dropout/StatefulPartitionedCall.uwiz_bernoulli_dropout/StatefulPartitionedCall2d
0uwiz_bernoulli_dropout_1/StatefulPartitionedCall0uwiz_bernoulli_dropout_1/StatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
А
D
(__inference_flatten_layer_call_fn_377597

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         АH* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3769942
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         АH2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╣
Ъ
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_377572

inputs 
cond_readvariableop_resource
identityИвcond
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
: *
dtype0
2
cond/ReadVariableOp╨
condIfcond/ReadVariableOp:value:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_377551*.
output_shapes
:         @*#
then_branchR
cond_true_3775502
conds
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:         @2
cond/Identityy
IdentityIdentitycond/Identity:output:0^cond*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         @:2
condcond:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: 
│'
О
F__inference_sequential_layer_call_and_return_conditional_losses_377251

inputs
conv2d_377223
conv2d_377225
conv2d_1_377228
conv2d_1_377230!
uwiz_bernoulli_dropout_377234
dense_377238
dense_377240
dense_1_377245
dense_1_377247
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв.uwiz_bernoulli_dropout/StatefulPartitionedCallв0uwiz_bernoulli_dropout_1/StatefulPartitionedCallЁ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_377223conv2d_377225*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3768532 
conv2d/StatefulPartitionedCallЫ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_377228conv2d_1_377230*
Tin
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3768752"
 conv2d_1/StatefulPartitionedCallь
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         @* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3768912
max_pooling2d/PartitionedCall┐
.uwiz_bernoulli_dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0uwiz_bernoulli_dropout_377234*
Tin
2*
Tout
2*/
_output_shapes
:         @*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*[
fVRT
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_37696920
.uwiz_bernoulli_dropout/StatefulPartitionedCallс
flatten/PartitionedCallPartitionedCall7uwiz_bernoulli_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         АH* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3769942
flatten/PartitionedCall■
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_377238dense_377240*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3770132
dense/StatefulPartitionedCall╛
0uwiz_bernoulli_dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0uwiz_bernoulli_dropout_377234*
Tin
2*
Tout
2*(
_output_shapes
:         А*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*]
fXRV
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_37708622
0uwiz_bernoulli_dropout_1/StatefulPartitionedCallа
dense_1/StatefulPartitionedCallStatefulPartitionedCall9uwiz_bernoulli_dropout_1/StatefulPartitionedCall:output:0dense_1_377245dense_1_377247*
Tin
2*
Tout
2*'
_output_shapes
:         
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3771152!
dense_1/StatefulPartitionedCallц
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall/^uwiz_bernoulli_dropout/StatefulPartitionedCall1^uwiz_bernoulli_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.uwiz_bernoulli_dropout/StatefulPartitionedCall.uwiz_bernoulli_dropout/StatefulPartitionedCall2d
0uwiz_bernoulli_dropout_1/StatefulPartitionedCall0uwiz_bernoulli_dropout_1/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
╣
Ъ
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_377547

inputs 
cond_readvariableop_resource
identityИвcond
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
: *
dtype0
2
cond/ReadVariableOp╨
condIfcond/ReadVariableOp:value:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_377519*.
output_shapes
:         @*#
then_branchR
cond_true_3775182
conds
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:         @2
cond/Identityy
IdentityIdentitycond/Identity:output:0^cond*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         @:2
condcond:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: 
С
;
cond_true_377518
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/ConstЗ
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
▄

9__inference_uwiz_bernoulli_dropout_1_layer_call_fn_377681

inputs
unknown
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:         А*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*]
fXRV
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_3770612
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: 
С
Ь
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_377649

inputs 
cond_readvariableop_resource
identityИвcond
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
: *
dtype0
2
cond/ReadVariableOp┬
condIfcond/ReadVariableOp:value:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:         А* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_377621*'
output_shapes
:         А*#
then_branchR
cond_true_3776202
condl
cond/IdentityIdentitycond:output:0*
T0*(
_output_shapes
:         А2
cond/Identityr
IdentityIdentitycond/Identity:output:0^cond*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А:2
condcond:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: 
ёФ
ь
"__inference__traced_restore_377945
file_prefix
assignvariableop_variable&
"assignvariableop_1_conv2d_2_kernel$
 assignvariableop_2_conv2d_2_bias(
$assignvariableop_3_conv2d_1_1_kernel&
"assignvariableop_4_conv2d_1_1_bias%
!assignvariableop_5_dense_2_kernel#
assignvariableop_6_dense_2_bias'
#assignvariableop_7_dense_1_1_kernel%
!assignvariableop_8_dense_1_1_bias$
 assignvariableop_9_adadelta_iter&
"assignvariableop_10_adadelta_decay.
*assignvariableop_11_adadelta_learning_rate$
 assignvariableop_12_adadelta_rho
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1;
7assignvariableop_17_adadelta_conv2d_2_kernel_accum_grad9
5assignvariableop_18_adadelta_conv2d_2_bias_accum_grad=
9assignvariableop_19_adadelta_conv2d_1_1_kernel_accum_grad;
7assignvariableop_20_adadelta_conv2d_1_1_bias_accum_grad:
6assignvariableop_21_adadelta_dense_2_kernel_accum_grad8
4assignvariableop_22_adadelta_dense_2_bias_accum_grad<
8assignvariableop_23_adadelta_dense_1_1_kernel_accum_grad:
6assignvariableop_24_adadelta_dense_1_1_bias_accum_grad:
6assignvariableop_25_adadelta_conv2d_2_kernel_accum_var8
4assignvariableop_26_adadelta_conv2d_2_bias_accum_var<
8assignvariableop_27_adadelta_conv2d_1_1_kernel_accum_var:
6assignvariableop_28_adadelta_conv2d_1_1_bias_accum_var9
5assignvariableop_29_adadelta_dense_2_kernel_accum_var7
3assignvariableop_30_adadelta_dense_2_bias_accum_var;
7assignvariableop_31_adadelta_dense_1_1_kernel_accum_var9
5assignvariableop_32_adadelta_dense_1_1_bias_accum_var
identity_34ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1╣
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*┼
value╗B╕!B1stochastic_mode_tensor/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names╨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╙
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ъ
_output_shapesЗ
Д:::::::::::::::::::::::::::::::::*/
dtypes%
#2!
	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0
*
_output_shapes
:2

IdentityЙ
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0*
_output_shapes
 *
dtype0
2
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ш
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_2_kernelIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ц
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_2_biasIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ъ
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv2d_1_1_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ш
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_1_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ч
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_2_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Х
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_2_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Щ
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_1_1_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ч
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_1_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0	*
_output_shapes
:2

Identity_9Ц
AssignVariableOp_9AssignVariableOp assignvariableop_9_adadelta_iterIdentity_9:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Ы
AssignVariableOp_10AssignVariableOp"assignvariableop_10_adadelta_decayIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11г
AssignVariableOp_11AssignVariableOp*assignvariableop_11_adadelta_learning_rateIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Щ
AssignVariableOp_12AssignVariableOp assignvariableop_12_adadelta_rhoIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Т
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Т
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ф
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Ф
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17░
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adadelta_conv2d_2_kernel_accum_gradIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18о
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adadelta_conv2d_2_bias_accum_gradIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19▓
AssignVariableOp_19AssignVariableOp9assignvariableop_19_adadelta_conv2d_1_1_kernel_accum_gradIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20░
AssignVariableOp_20AssignVariableOp7assignvariableop_20_adadelta_conv2d_1_1_bias_accum_gradIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21п
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adadelta_dense_2_kernel_accum_gradIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22н
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adadelta_dense_2_bias_accum_gradIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23▒
AssignVariableOp_23AssignVariableOp8assignvariableop_23_adadelta_dense_1_1_kernel_accum_gradIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24п
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adadelta_dense_1_1_bias_accum_gradIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25п
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adadelta_conv2d_2_kernel_accum_varIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26н
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adadelta_conv2d_2_bias_accum_varIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27▒
AssignVariableOp_27AssignVariableOp8assignvariableop_27_adadelta_conv2d_1_1_kernel_accum_varIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28п
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adadelta_conv2d_1_1_bias_accum_varIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29о
AssignVariableOp_29AssignVariableOp5assignvariableop_29_adadelta_dense_2_kernel_accum_varIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30м
AssignVariableOp_30AssignVariableOp3assignvariableop_30_adadelta_dense_2_bias_accum_varIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31░
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adadelta_dense_1_1_kernel_accum_varIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32о
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adadelta_dense_1_1_bias_accum_varIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp┤
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33┴
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*Ы
_input_shapesЙ
Ж: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: 
╣
Ъ
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_376944

inputs 
cond_readvariableop_resource
identityИвcond
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
: *
dtype0
2
cond/ReadVariableOp╨
condIfcond/ReadVariableOp:value:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_376916*.
output_shapes
:         @*#
then_branchR
cond_true_3769152
conds
cond/IdentityIdentitycond:output:0*
T0*/
_output_shapes
:         @2
cond/Identityy
IdentityIdentitycond/Identity:output:0^cond*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         @:2
condcond:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: 
¤

X
)uwiz_bernoulli_dropout_1_cond_true_377358
dropout_mul_dense_relu
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstД
dropout/MulMuldropout_mul_dense_reludropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Muld
dropout/ShapeShapedropout_mul_dense_relu*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
╪

;
cond_true_377652
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstА
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
Ї
}
7__inference_uwiz_bernoulli_dropout_layer_call_fn_377586

inputs
unknown
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*/
_output_shapes
:         @*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*[
fVRT
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_3769692
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         @:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: 
Ў
}
(__inference_dense_1_layer_call_fn_377708

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3771152
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
д
8
cond_false_376948
identity_inputs

identity_1k
IdentityIdentityidentity_inputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
д
8
cond_false_377551
identity_inputs

identity_1k
IdentityIdentityidentity_inputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
Х5
╝
F__inference_sequential_layer_call_and_return_conditional_losses_377393

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource7
3uwiz_bernoulli_dropout_cond_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИвuwiz_bernoulli_dropout/condвuwiz_bernoulli_dropout_1/condк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╣
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d/Relu░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╥
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Relu├
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool─
*uwiz_bernoulli_dropout/cond/ReadVariableOpReadVariableOp3uwiz_bernoulli_dropout_cond_readvariableop_resource*
_output_shapes
: *
dtype0
2,
*uwiz_bernoulli_dropout/cond/ReadVariableOp█
uwiz_bernoulli_dropout/condIf2uwiz_bernoulli_dropout/cond/ReadVariableOp:value:0max_pooling2d/MaxPool:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *;
else_branch,R*
(uwiz_bernoulli_dropout_cond_false_377322*.
output_shapes
:         @*:
then_branch+R)
'uwiz_bernoulli_dropout_cond_true_3773212
uwiz_bernoulli_dropout/cond╕
$uwiz_bernoulli_dropout/cond/IdentityIdentity$uwiz_bernoulli_dropout/cond:output:0*
T0*/
_output_shapes
:         @2&
$uwiz_bernoulli_dropout/cond/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
flatten/Constз
flatten/ReshapeReshape-uwiz_bernoulli_dropout/cond/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         АH2
flatten/Reshapeб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2

dense/Relu╚
,uwiz_bernoulli_dropout_1/cond/ReadVariableOpReadVariableOp3uwiz_bernoulli_dropout_cond_readvariableop_resource*
_output_shapes
: *
dtype0
2.
,uwiz_bernoulli_dropout_1/cond/ReadVariableOpя
uwiz_bernoulli_dropout_1/condIf4uwiz_bernoulli_dropout_1/cond/ReadVariableOp:value:0dense/Relu:activations:0^uwiz_bernoulli_dropout/cond*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:         А* 
_read_only_resource_inputs
 *=
else_branch.R,
*uwiz_bernoulli_dropout_1_cond_false_377359*'
output_shapes
:         А*<
then_branch-R+
)uwiz_bernoulli_dropout_1_cond_true_3773582
uwiz_bernoulli_dropout_1/cond╖
&uwiz_bernoulli_dropout_1/cond/IdentityIdentity&uwiz_bernoulli_dropout_1/cond:output:0*
T0*(
_output_shapes
:         А2(
&uwiz_bernoulli_dropout_1/cond/Identityж
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
dense_1/MatMul/ReadVariableOp┤
dense_1/MatMulMatMul/uwiz_bernoulli_dropout_1/cond/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense_1/Softmaxл
IdentityIdentitydense_1/Softmax:softmax:0^uwiz_bernoulli_dropout/cond^uwiz_bernoulli_dropout_1/cond*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::2:
uwiz_bernoulli_dropout/conduwiz_bernoulli_dropout/cond2>
uwiz_bernoulli_dropout_1/conduwiz_bernoulli_dropout_1/cond:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
┘

<
cond_false_377033
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstА
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
Б
w
2sequential_uwiz_bernoulli_dropout_cond_true_3767830
,dropout_mul_sequential_max_pooling2d_maxpool
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constб
dropout/MulMul,dropout_mul_sequential_max_pooling2d_maxpooldropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mulz
dropout/ShapeShape,dropout_mul_sequential_max_pooling2d_maxpool*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
╪

;
cond_true_377064
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstА
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
С
;
cond_true_377550
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/ConstЗ
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
╧

я
+__inference_sequential_layer_call_fn_377272
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityИвStatefulPartitionedCall╡
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*'
_output_shapes
:         
*+
_read_only_resource_inputs
		**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3772512
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
╓
b
(uwiz_bernoulli_dropout_cond_false_377322%
!dropout_mul_max_pooling2d_maxpool
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/ConstЦ
dropout/MulMul!dropout_mul_max_pooling2d_maxpooldropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mulo
dropout/ShapeShape!dropout_mul_max_pooling2d_maxpool*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
╧

я
+__inference_sequential_layer_call_fn_377218
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityИвStatefulPartitionedCall╡
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*'
_output_shapes
:         
*+
_read_only_resource_inputs
		**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3771972
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
╩
k
5sequential_uwiz_bernoulli_dropout_1_cond_false_376814"
identity_sequential_dense_relu

identity_1s
IdentityIdentityidentity_sequential_dense_relu*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
│'
О
F__inference_sequential_layer_call_and_return_conditional_losses_377197

inputs
conv2d_377169
conv2d_377171
conv2d_1_377174
conv2d_1_377176!
uwiz_bernoulli_dropout_377180
dense_377184
dense_377186
dense_1_377191
dense_1_377193
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв.uwiz_bernoulli_dropout/StatefulPartitionedCallв0uwiz_bernoulli_dropout_1/StatefulPartitionedCallЁ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_377169conv2d_377171*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3768532 
conv2d/StatefulPartitionedCallЫ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_377174conv2d_1_377176*
Tin
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3768752"
 conv2d_1/StatefulPartitionedCallь
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         @* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3768912
max_pooling2d/PartitionedCall┐
.uwiz_bernoulli_dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0uwiz_bernoulli_dropout_377180*
Tin
2*
Tout
2*/
_output_shapes
:         @*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*[
fVRT
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_37694420
.uwiz_bernoulli_dropout/StatefulPartitionedCallс
flatten/PartitionedCallPartitionedCall7uwiz_bernoulli_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         АH* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3769942
flatten/PartitionedCall■
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_377184dense_377186*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3770132
dense/StatefulPartitionedCall╛
0uwiz_bernoulli_dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0uwiz_bernoulli_dropout_377180*
Tin
2*
Tout
2*(
_output_shapes
:         А*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*]
fXRV
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_37706122
0uwiz_bernoulli_dropout_1/StatefulPartitionedCallа
dense_1/StatefulPartitionedCallStatefulPartitionedCall9uwiz_bernoulli_dropout_1/StatefulPartitionedCall:output:0dense_1_377191dense_1_377193*
Tin
2*
Tout
2*'
_output_shapes
:         
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3771152!
dense_1/StatefulPartitionedCallц
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall/^uwiz_bernoulli_dropout/StatefulPartitionedCall1^uwiz_bernoulli_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.uwiz_bernoulli_dropout/StatefulPartitionedCall.uwiz_bernoulli_dropout/StatefulPartitionedCall2d
0uwiz_bernoulli_dropout_1/StatefulPartitionedCall0uwiz_bernoulli_dropout_1/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
З
J
.__inference_max_pooling2d_layer_call_fn_376897

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3768912
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
С
;
cond_true_376915
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/ConstЗ
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
┘
^
(uwiz_bernoulli_dropout_cond_false_377412"
identity_max_pooling2d_maxpool

identity_1z
IdentityIdentityidentity_max_pooling2d_maxpool*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
¤

X
)uwiz_bernoulli_dropout_1_cond_true_377441
dropout_mul_dense_relu
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstД
dropout/MulMuldropout_mul_dense_reludropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Muld
dropout/ShapeShapedropout_mul_dense_relu*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
╜

щ
+__inference_sequential_layer_call_fn_377492

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityИвStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*'
_output_shapes
:         
*+
_read_only_resource_inputs
		**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3771972
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
▄

9__inference_uwiz_bernoulli_dropout_1_layer_call_fn_377688

inputs
unknown
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:         А*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*]
fXRV
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_3770862
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: 
╛T
┤
__inference__traced_save_377834
file_prefix'
#savev2_variable_read_readvariableop
.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop0
,savev2_conv2d_1_1_kernel_read_readvariableop.
*savev2_conv2d_1_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop/
+savev2_dense_1_1_kernel_read_readvariableop-
)savev2_dense_1_1_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopB
>savev2_adadelta_conv2d_2_kernel_accum_grad_read_readvariableop@
<savev2_adadelta_conv2d_2_bias_accum_grad_read_readvariableopD
@savev2_adadelta_conv2d_1_1_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_conv2d_1_1_bias_accum_grad_read_readvariableopA
=savev2_adadelta_dense_2_kernel_accum_grad_read_readvariableop?
;savev2_adadelta_dense_2_bias_accum_grad_read_readvariableopC
?savev2_adadelta_dense_1_1_kernel_accum_grad_read_readvariableopA
=savev2_adadelta_dense_1_1_bias_accum_grad_read_readvariableopA
=savev2_adadelta_conv2d_2_kernel_accum_var_read_readvariableop?
;savev2_adadelta_conv2d_2_bias_accum_var_read_readvariableopC
?savev2_adadelta_conv2d_1_1_kernel_accum_var_read_readvariableopA
=savev2_adadelta_conv2d_1_1_bias_accum_var_read_readvariableop@
<savev2_adadelta_dense_2_kernel_accum_var_read_readvariableop>
:savev2_adadelta_dense_2_bias_accum_var_read_readvariableopB
>savev2_adadelta_dense_1_1_kernel_accum_var_read_readvariableop@
<savev2_adadelta_dense_1_1_bias_accum_var_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1П
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f5daefa73a6c4899809c5efa592926bb/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename│
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*┼
value╗B╕!B1stochastic_mode_tensor/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names╩
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices·
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop,savev2_conv2d_1_1_kernel_read_readvariableop*savev2_conv2d_1_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop+savev2_dense_1_1_kernel_read_readvariableop)savev2_dense_1_1_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adadelta_conv2d_2_kernel_accum_grad_read_readvariableop<savev2_adadelta_conv2d_2_bias_accum_grad_read_readvariableop@savev2_adadelta_conv2d_1_1_kernel_accum_grad_read_readvariableop>savev2_adadelta_conv2d_1_1_bias_accum_grad_read_readvariableop=savev2_adadelta_dense_2_kernel_accum_grad_read_readvariableop;savev2_adadelta_dense_2_bias_accum_grad_read_readvariableop?savev2_adadelta_dense_1_1_kernel_accum_grad_read_readvariableop=savev2_adadelta_dense_1_1_bias_accum_grad_read_readvariableop=savev2_adadelta_conv2d_2_kernel_accum_var_read_readvariableop;savev2_adadelta_conv2d_2_bias_accum_var_read_readvariableop?savev2_adadelta_conv2d_1_1_kernel_accum_var_read_readvariableop=savev2_adadelta_conv2d_1_1_bias_accum_var_read_readvariableop<savev2_adadelta_dense_2_kernel_accum_var_read_readvariableop:savev2_adadelta_dense_2_bias_accum_var_read_readvariableop>savev2_adadelta_dense_1_1_kernel_accum_var_read_readvariableop<savev2_adadelta_dense_1_1_bias_accum_var_read_readvariableop"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!
	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*з
_input_shapesХ
Т: : : : : @:@:
АHА:А:	А
:
: : : : : : : : : : : @:@:
АHА:А:	А
:
: : : @:@:
АHА:А:	А
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
АHА:!

_output_shapes	
:А:%!

_output_shapes
:	А
: 	

_output_shapes
:
:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
АHА:!

_output_shapes	
:А:%!

_output_shapes
:	А
: 

_output_shapes
:
:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
АHА:!

_output_shapes	
:А:% !

_output_shapes
:	А
: !

_output_shapes
:
:"

_output_shapes
: 
И
8
cond_false_377065
identity_inputs

identity_1d
IdentityIdentityidentity_inputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
 
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_376891

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
я
л
C__inference_dense_1_layer_call_and_return_conditional_losses_377115

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Т
<
cond_false_377519
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/ConstЗ
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
Ї
}
7__inference_uwiz_bernoulli_dropout_layer_call_fn_377579

inputs
unknown
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*/
_output_shapes
:         @*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*[
fVRT
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_3769442
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         @:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: 
╜

щ
+__inference_sequential_layer_call_fn_377515

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identityИвStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*'
_output_shapes
:         
*+
_read_only_resource_inputs
		**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3772512
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
С
Ь
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_377061

inputs 
cond_readvariableop_resource
identityИвcond
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
: *
dtype0
2
cond/ReadVariableOp┬
condIfcond/ReadVariableOp:value:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*(
_output_shapes
:         А* 
_read_only_resource_inputs
 *$
else_branchR
cond_false_377033*'
output_shapes
:         А*#
then_branchR
cond_true_3770322
condl
cond/IdentityIdentitycond:output:0*
T0*(
_output_shapes
:         А2
cond/Identityr
IdentityIdentitycond/Identity:output:0^cond*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А:2
condcond:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: 
й
n
4sequential_uwiz_bernoulli_dropout_1_cond_true_376813%
!dropout_mul_sequential_dense_relu
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstП
dropout/MulMul!dropout_mul_sequential_dense_reludropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Mulo
dropout/ShapeShape!dropout_mul_sequential_dense_relu*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
│

м
D__inference_conv2d_1_layer_call_and_return_conditional_losses_376875

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
й
U
*uwiz_bernoulli_dropout_1_cond_false_377442
identity_dense_relu

identity_1h
IdentityIdentityidentity_dense_relu*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
Т
<
cond_false_376916
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/ConstЗ
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
я
л
C__inference_dense_1_layer_call_and_return_conditional_losses_377699

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ы
й
A__inference_dense_layer_call_and_return_conditional_losses_377608

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АHА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         АH:::P L
(
_output_shapes
:         АH
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╪

;
cond_true_377032
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstА
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
С
;
cond_true_376947
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/ConstЗ
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @
┌
|
'__inference_conv2d_layer_call_fn_376863

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3768532
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┼'
Ф
F__inference_sequential_layer_call_and_return_conditional_losses_377132
conv2d_input
conv2d_376901
conv2d_376903
conv2d_1_376906
conv2d_1_376908!
uwiz_bernoulli_dropout_376985
dense_377024
dense_377026
dense_1_377126
dense_1_377128
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв.uwiz_bernoulli_dropout/StatefulPartitionedCallв0uwiz_bernoulli_dropout_1/StatefulPartitionedCallЎ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_376901conv2d_376903*
Tin
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3768532 
conv2d/StatefulPartitionedCallЫ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_376906conv2d_1_376908*
Tin
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3768752"
 conv2d_1/StatefulPartitionedCallь
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         @* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3768912
max_pooling2d/PartitionedCall┐
.uwiz_bernoulli_dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0uwiz_bernoulli_dropout_376985*
Tin
2*
Tout
2*/
_output_shapes
:         @*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*[
fVRT
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_37694420
.uwiz_bernoulli_dropout/StatefulPartitionedCallс
flatten/PartitionedCallPartitionedCall7uwiz_bernoulli_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         АH* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3769942
flatten/PartitionedCall■
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_377024dense_377026*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3770132
dense/StatefulPartitionedCall╛
0uwiz_bernoulli_dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0uwiz_bernoulli_dropout_376985*
Tin
2*
Tout
2*(
_output_shapes
:         А*#
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*]
fXRV
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_37706122
0uwiz_bernoulli_dropout_1/StatefulPartitionedCallа
dense_1/StatefulPartitionedCallStatefulPartitionedCall9uwiz_bernoulli_dropout_1/StatefulPartitionedCall:output:0dense_1_377126dense_1_377128*
Tin
2*
Tout
2*'
_output_shapes
:         
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3771152!
dense_1/StatefulPartitionedCallц
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall/^uwiz_bernoulli_dropout/StatefulPartitionedCall1^uwiz_bernoulli_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:         :::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2`
.uwiz_bernoulli_dropout/StatefulPartitionedCall.uwiz_bernoulli_dropout/StatefulPartitionedCall2d
0uwiz_bernoulli_dropout_1/StatefulPartitionedCall0uwiz_bernoulli_dropout_1/StatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameconv2d_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
И
8
cond_false_377653
identity_inputs

identity_1d
IdentityIdentityidentity_inputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
╗
_
C__inference_flatten_layer_call_and_return_conditional_losses_376994

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     $  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АH2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         АH2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╪

;
cond_true_377620
dropout_mul_inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/ConstА
dropout/MulMuldropout_mul_inputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/Mul`
dropout/ShapeShapedropout_mul_inputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:. *
(
_output_shapes
:         А
√
t
3sequential_uwiz_bernoulli_dropout_cond_false_376784-
)identity_sequential_max_pooling2d_maxpool

identity_1Е
IdentityIdentity)identity_sequential_max_pooling2d_maxpool*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:5 1
/
_output_shapes
:         @"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╝
serving_defaultи
M
conv2d_input=
serving_default_conv2d_input:0         ;
dense_10
StatefulPartitionedCall:0         
tensorflow/serving/predict:ЦГ
Щ;
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	stochastic_mode_tensor
	_stochastic_mode_tensor

	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+Д&call_and_return_all_conditional_losses
Е_default_save_signature
Ж__call__"┬7
_tf_keras_sequentialг7{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "UwizBernoulliDropout", "config": {"name": "UwBernoulliDropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "UwizBernoulliDropout", "config": {"name": "UwBernoulliDropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "UwizBernoulliDropout", "config": {"name": "UwBernoulliDropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "UwizBernoulliDropout", "config": {"name": "UwBernoulliDropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adadelta", "config": {"name": "Adadelta", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.949999988079071, "epsilon": 1e-07}}}}
└


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+З&call_and_return_all_conditional_losses
И__call__"Щ	
_tf_keras_layer {"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
┼	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"Ю
_tf_keras_layerД{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26, 26, 32]}}
┌
	variables
regularization_losses
trainable_variables
	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"╔
_tf_keras_layerп{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"╫
_tf_keras_layer╜{"class_name": "UwizBernoulliDropout", "name": "uwiz_bernoulli_dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "UwBernoulliDropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
┴
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"░
_tf_keras_layerЦ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
╨

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"й
_tf_keras_layerП{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9216}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9216]}}
щ
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"╪
_tf_keras_layer╛{"class_name": "UwizBernoulliDropout", "name": "uwiz_bernoulli_dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "UwBernoulliDropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
╘

2kernel
3bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"н
_tf_keras_layerУ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
:
 2Variable
р
8iter
	9decay
:learning_rate
;rho
accum_gradt
accum_gradu
accum_gradv
accum_gradw(
accum_gradx)
accum_grady2
accum_gradz3
accum_grad{	accum_var|	accum_var}	accum_var~	accum_var(	accum_varА)	accum_varБ2	accum_varВ3	accum_varГ"
	optimizer
_
0
1
2
3
(4
)5
26
37
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
(4
)5
26
37"
trackable_list_wrapper
╬
<layer_metrics

=layers
>metrics
?non_trainable_variables
	variables
regularization_losses
trainable_variables
@layer_regularization_losses
Ж__call__
Е_default_save_signature
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
-
Чserving_default"
signature_map
):' 2conv2d_2/kernel
: 2conv2d_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Alayer_metrics

Blayers
Cmetrics
Dnon_trainable_variables
	variables
regularization_losses
trainable_variables
Elayer_regularization_losses
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
+:) @2conv2d_1_1/kernel
:@2conv2d_1_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Flayer_metrics

Glayers
Hmetrics
Inon_trainable_variables
	variables
regularization_losses
trainable_variables
Jlayer_regularization_losses
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Klayer_metrics

Llayers
Mmetrics
Nnon_trainable_variables
	variables
regularization_losses
trainable_variables
Olayer_regularization_losses
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Player_metrics

Qlayers
Rmetrics
Snon_trainable_variables
 	variables
!regularization_losses
"trainable_variables
Tlayer_regularization_losses
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Ulayer_metrics

Vlayers
Wmetrics
Xnon_trainable_variables
$	variables
%regularization_losses
&trainable_variables
Ylayer_regularization_losses
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
": 
АHА2dense_2/kernel
:А2dense_2/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
░
Zlayer_metrics

[layers
\metrics
]non_trainable_variables
*	variables
+regularization_losses
,trainable_variables
^layer_regularization_losses
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
_layer_metrics

`layers
ametrics
bnon_trainable_variables
.	variables
/regularization_losses
0trainable_variables
clayer_regularization_losses
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
#:!	А
2dense_1_1/kernel
:
2dense_1_1/bias
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
░
dlayer_metrics

elayers
fmetrics
gnon_trainable_variables
4	variables
5regularization_losses
6trainable_variables
hlayer_regularization_losses
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╗
	ktotal
	lcount
m	variables
n	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 
	ototal
	pcount
q
_fn_kwargs
r	variables
s	keras_api"╕
_tf_keras_metricЭ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
k0
l1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
o0
p1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
;:9 2#Adadelta/conv2d_2/kernel/accum_grad
-:+ 2!Adadelta/conv2d_2/bias/accum_grad
=:; @2%Adadelta/conv2d_1_1/kernel/accum_grad
/:-@2#Adadelta/conv2d_1_1/bias/accum_grad
4:2
АHА2"Adadelta/dense_2/kernel/accum_grad
-:+А2 Adadelta/dense_2/bias/accum_grad
5:3	А
2$Adadelta/dense_1_1/kernel/accum_grad
.:,
2"Adadelta/dense_1_1/bias/accum_grad
::8 2"Adadelta/conv2d_2/kernel/accum_var
,:* 2 Adadelta/conv2d_2/bias/accum_var
<:: @2$Adadelta/conv2d_1_1/kernel/accum_var
.:,@2"Adadelta/conv2d_1_1/bias/accum_var
3:1
АHА2!Adadelta/dense_2/kernel/accum_var
,:*А2Adadelta/dense_2/bias/accum_var
4:2	А
2#Adadelta/dense_1_1/kernel/accum_var
-:+
2!Adadelta/dense_1_1/bias/accum_var
ц2у
F__inference_sequential_layer_call_and_return_conditional_losses_377132
F__inference_sequential_layer_call_and_return_conditional_losses_377163
F__inference_sequential_layer_call_and_return_conditional_losses_377393
F__inference_sequential_layer_call_and_return_conditional_losses_377469└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ь2щ
!__inference__wrapped_model_376841├
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+
conv2d_input         
·2ў
+__inference_sequential_layer_call_fn_377492
+__inference_sequential_layer_call_fn_377218
+__inference_sequential_layer_call_fn_377515
+__inference_sequential_layer_call_fn_377272└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
б2Ю
B__inference_conv2d_layer_call_and_return_conditional_losses_376853╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Ж2Г
'__inference_conv2d_layer_call_fn_376863╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
г2а
D__inference_conv2d_1_layer_call_and_return_conditional_losses_376875╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
И2Е
)__inference_conv2d_1_layer_call_fn_376885╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
▒2о
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_376891р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ц2У
.__inference_max_pooling2d_layer_call_fn_376897р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
т2▀
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_377572
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_377547┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
м2й
7__inference_uwiz_bernoulli_dropout_layer_call_fn_377586
7__inference_uwiz_bernoulli_dropout_layer_call_fn_377579┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_377592в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_flatten_layer_call_fn_377597в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_layer_call_and_return_conditional_losses_377608в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_layer_call_fn_377617в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ц2у
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_377674
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_377649┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
░2н
9__inference_uwiz_bernoulli_dropout_1_layer_call_fn_377681
9__inference_uwiz_bernoulli_dropout_1_layer_call_fn_377688┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_377699в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_1_layer_call_fn_377708в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
8B6
$__inference_signature_wrapper_377303conv2d_inputв
!__inference__wrapped_model_376841}		()23=в:
3в0
.К+
conv2d_input         
к "1к.
,
dense_1!К
dense_1         
┘
D__inference_conv2d_1_layer_call_and_return_conditional_losses_376875РIвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                           @
Ъ ▒
)__inference_conv2d_1_layer_call_fn_376885ГIвF
?в<
:К7
inputs+                            
к "2К/+                           @╫
B__inference_conv2d_layer_call_and_return_conditional_losses_376853РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                            
Ъ п
'__inference_conv2d_layer_call_fn_376863ГIвF
?в<
:К7
inputs+                           
к "2К/+                            д
C__inference_dense_1_layer_call_and_return_conditional_losses_377699]230в-
&в#
!К
inputs         А
к "%в"
К
0         

Ъ |
(__inference_dense_1_layer_call_fn_377708P230в-
&в#
!К
inputs         А
к "К         
г
A__inference_dense_layer_call_and_return_conditional_losses_377608^()0в-
&в#
!К
inputs         АH
к "&в#
К
0         А
Ъ {
&__inference_dense_layer_call_fn_377617Q()0в-
&в#
!К
inputs         АH
к "К         Аи
C__inference_flatten_layer_call_and_return_conditional_losses_377592a7в4
-в*
(К%
inputs         @
к "&в#
К
0         АH
Ъ А
(__inference_flatten_layer_call_fn_377597T7в4
-в*
(К%
inputs         @
к "К         АHь
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_376891ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ─
.__inference_max_pooling2d_layer_call_fn_376897СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ├
F__inference_sequential_layer_call_and_return_conditional_losses_377132y		()23EвB
;в8
.К+
conv2d_input         
p

 
к "%в"
К
0         

Ъ ├
F__inference_sequential_layer_call_and_return_conditional_losses_377163y		()23EвB
;в8
.К+
conv2d_input         
p 

 
к "%в"
К
0         

Ъ ╜
F__inference_sequential_layer_call_and_return_conditional_losses_377393s		()23?в<
5в2
(К%
inputs         
p

 
к "%в"
К
0         

Ъ ╜
F__inference_sequential_layer_call_and_return_conditional_losses_377469s		()23?в<
5в2
(К%
inputs         
p 

 
к "%в"
К
0         

Ъ Ы
+__inference_sequential_layer_call_fn_377218l		()23EвB
;в8
.К+
conv2d_input         
p

 
к "К         
Ы
+__inference_sequential_layer_call_fn_377272l		()23EвB
;в8
.К+
conv2d_input         
p 

 
к "К         
Х
+__inference_sequential_layer_call_fn_377492f		()23?в<
5в2
(К%
inputs         
p

 
к "К         
Х
+__inference_sequential_layer_call_fn_377515f		()23?в<
5в2
(К%
inputs         
p 

 
к "К         
╢
$__inference_signature_wrapper_377303Н		()23MвJ
в 
Cк@
>
conv2d_input.К+
conv2d_input         "1к.
,
dense_1!К
dense_1         
╣
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_377649a	4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ ╣
T__inference_uwiz_bernoulli_dropout_1_layer_call_and_return_conditional_losses_377674a	4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ С
9__inference_uwiz_bernoulli_dropout_1_layer_call_fn_377681T	4в1
*в'
!К
inputs         А
p
к "К         АС
9__inference_uwiz_bernoulli_dropout_1_layer_call_fn_377688T	4в1
*в'
!К
inputs         А
p 
к "К         А┼
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_377547o	;в8
1в.
(К%
inputs         @
p
к "-в*
#К 
0         @
Ъ ┼
R__inference_uwiz_bernoulli_dropout_layer_call_and_return_conditional_losses_377572o	;в8
1в.
(К%
inputs         @
p 
к "-в*
#К 
0         @
Ъ Э
7__inference_uwiz_bernoulli_dropout_layer_call_fn_377579b	;в8
1в.
(К%
inputs         @
p
к " К         @Э
7__inference_uwiz_bernoulli_dropout_layer_call_fn_377586b	;в8
1в.
(К%
inputs         @
p 
к " К         @