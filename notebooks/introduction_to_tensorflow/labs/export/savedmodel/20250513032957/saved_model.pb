��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
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
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
�
Adam/v/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_24/bias
y
(Adam/v/dense_24/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_24/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_24/bias
y
(Adam/m/dense_24/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_24/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_24/kernel
�
*Adam/v/dense_24/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_24/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_24/kernel
�
*Adam/m/dense_24/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_24/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_23/bias
y
(Adam/v/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_23/bias
y
(Adam/m/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_23/kernel
�
*Adam/v/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_23/kernel
�
*Adam/m/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_22/bias
z
(Adam/v/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_22/bias
z
(Adam/m/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/v/dense_22/kernel
�
*Adam/v/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*'
shared_nameAdam/m/dense_22/kernel
�
*Adam/m/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/v/dense_21/bias
z
(Adam/v/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/m/dense_21/bias
z
(Adam/m/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_21/kernel
�
*Adam/v/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_21/kernel
�
*Adam/m/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:*
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:*
dtype0
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	�*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:�*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
��*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:�*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	�*
dtype0
�
serving_default_dense_21_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_21_inputdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_846517

NoOpNoOp
�7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�7
value�6B�6 B�6
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
<
0
1
2
3
$4
%5
,6
-7*
<
0
1
2
3
$4
%5
,6
-7*
* 
�
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
3trace_0
4trace_1
5trace_2
6trace_3* 
6
7trace_0
8trace_1
9trace_2
:trace_3* 
* 
�
;
_variables
<_iterations
=_learning_rate
>_index_dict
?
_momentums
@_velocities
A_update_step_xla*

Bserving_default* 

0
1*

0
1*
* 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Htrace_0* 

Itrace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Otrace_0* 

Ptrace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Vtrace_0* 

Wtrace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
_Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_24/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

_0
`1
a2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
<0
b1
c2
d3
e4
f5
g6
h7
i8
j9
k10
l11
m12
n13
o14
p15
q16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
b0
d1
f2
h3
j4
l5
n6
p7*
<
c0
e1
g2
i3
k4
m5
o6
q7*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
r	variables
s	keras_api
	ttotal
	ucount*
H
v	variables
w	keras_api
	xtotal
	ycount
z
_fn_kwargs*
H
{	variables
|	keras_api
	}total
	~count

_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/dense_21/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_21/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_21/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_21/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_22/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_22/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_22/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_22/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_23/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_23/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_23/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_23/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_24/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_24/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_24/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_24/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*

t0
u1*

r	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

v	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

}0
~1*

{	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias	iterationlearning_rateAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biasAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biasAdam/m/dense_23/kernelAdam/v/dense_23/kernelAdam/m/dense_23/biasAdam/v/dense_23/biasAdam/m/dense_24/kernelAdam/v/dense_24/kernelAdam/m/dense_24/biasAdam/v/dense_24/biastotal_2count_2total_1count_1totalcountConst*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_846915
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias	iterationlearning_rateAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biasAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biasAdam/m/dense_23/kernelAdam/v/dense_23/kernelAdam/m/dense_23/biasAdam/v/dense_23/biasAdam/m/dense_24/kernelAdam/v/dense_24/kernelAdam/m/dense_24/biasAdam/v/dense_24/biastotal_2count_2total_1count_1totalcount*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_847021��
�	
�
-__inference_sequential_5_layer_call_fn_846538

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_846346o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
-__inference_sequential_5_layer_call_fn_846410
dense_21_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_846391o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�	
�
$__inference_signature_wrapper_846517
dense_21_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_846223o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�

�
D__inference_dense_22_layer_call_and_return_conditional_losses_846255

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_847021
file_prefix3
 assignvariableop_dense_21_kernel:	�/
 assignvariableop_1_dense_21_bias:	�6
"assignvariableop_2_dense_22_kernel:
��/
 assignvariableop_3_dense_22_bias:	�5
"assignvariableop_4_dense_23_kernel:	�.
 assignvariableop_5_dense_23_bias:4
"assignvariableop_6_dense_24_kernel:.
 assignvariableop_7_dense_24_bias:&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: =
*assignvariableop_10_adam_m_dense_21_kernel:	�=
*assignvariableop_11_adam_v_dense_21_kernel:	�7
(assignvariableop_12_adam_m_dense_21_bias:	�7
(assignvariableop_13_adam_v_dense_21_bias:	�>
*assignvariableop_14_adam_m_dense_22_kernel:
��>
*assignvariableop_15_adam_v_dense_22_kernel:
��7
(assignvariableop_16_adam_m_dense_22_bias:	�7
(assignvariableop_17_adam_v_dense_22_bias:	�=
*assignvariableop_18_adam_m_dense_23_kernel:	�=
*assignvariableop_19_adam_v_dense_23_kernel:	�6
(assignvariableop_20_adam_m_dense_23_bias:6
(assignvariableop_21_adam_v_dense_23_bias:<
*assignvariableop_22_adam_m_dense_24_kernel:<
*assignvariableop_23_adam_v_dense_24_kernel:6
(assignvariableop_24_adam_m_dense_24_bias:6
(assignvariableop_25_adam_v_dense_24_bias:%
assignvariableop_26_total_2: %
assignvariableop_27_count_2: %
assignvariableop_28_total_1: %
assignvariableop_29_count_1: #
assignvariableop_30_total: #
assignvariableop_31_count: 
identity_33��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_21_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_21_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_22_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_22_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_23_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_23_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_24_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_24_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp*assignvariableop_10_adam_m_dense_21_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp*assignvariableop_11_adam_v_dense_21_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp(assignvariableop_12_adam_m_dense_21_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_v_dense_21_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_m_dense_22_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_v_dense_22_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_m_dense_22_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_v_dense_22_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_m_dense_23_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_v_dense_23_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_m_dense_23_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_v_dense_23_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_m_dense_24_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_v_dense_24_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_m_dense_24_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_v_dense_24_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_2Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_2Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_1Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_1Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_countIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_33IdentityIdentity_32:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_33Identity_33:output:0*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846346

inputs"
dense_21_846325:	�
dense_21_846327:	�#
dense_22_846330:
��
dense_22_846332:	�"
dense_23_846335:	�
dense_23_846337:!
dense_24_846340:
dense_24_846342:
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinputsdense_21_846325dense_21_846327*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_846238�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_846330dense_22_846332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_846255�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_846335dense_23_846337*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_846272�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_846340dense_24_846342*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_846288x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
-__inference_sequential_5_layer_call_fn_846365
dense_21_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_846346o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�

�
D__inference_dense_21_layer_call_and_return_conditional_losses_846641

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
__inference__traced_save_846915
file_prefix9
&read_disablecopyonread_dense_21_kernel:	�5
&read_1_disablecopyonread_dense_21_bias:	�<
(read_2_disablecopyonread_dense_22_kernel:
��5
&read_3_disablecopyonread_dense_22_bias:	�;
(read_4_disablecopyonread_dense_23_kernel:	�4
&read_5_disablecopyonread_dense_23_bias::
(read_6_disablecopyonread_dense_24_kernel:4
&read_7_disablecopyonread_dense_24_bias:,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: C
0read_10_disablecopyonread_adam_m_dense_21_kernel:	�C
0read_11_disablecopyonread_adam_v_dense_21_kernel:	�=
.read_12_disablecopyonread_adam_m_dense_21_bias:	�=
.read_13_disablecopyonread_adam_v_dense_21_bias:	�D
0read_14_disablecopyonread_adam_m_dense_22_kernel:
��D
0read_15_disablecopyonread_adam_v_dense_22_kernel:
��=
.read_16_disablecopyonread_adam_m_dense_22_bias:	�=
.read_17_disablecopyonread_adam_v_dense_22_bias:	�C
0read_18_disablecopyonread_adam_m_dense_23_kernel:	�C
0read_19_disablecopyonread_adam_v_dense_23_kernel:	�<
.read_20_disablecopyonread_adam_m_dense_23_bias:<
.read_21_disablecopyonread_adam_v_dense_23_bias:B
0read_22_disablecopyonread_adam_m_dense_24_kernel:B
0read_23_disablecopyonread_adam_v_dense_24_kernel:<
.read_24_disablecopyonread_adam_m_dense_24_bias:<
.read_25_disablecopyonread_adam_v_dense_24_bias:+
!read_26_disablecopyonread_total_2: +
!read_27_disablecopyonread_count_2: +
!read_28_disablecopyonread_total_1: +
!read_29_disablecopyonread_count_1: )
read_30_disablecopyonread_total: )
read_31_disablecopyonread_count: 
savev2_const
identity_65��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_21_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_21_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_22_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_22_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_23_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_23_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_dense_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_dense_24_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_dense_24_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_dense_24_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead0read_10_disablecopyonread_adam_m_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp0read_10_disablecopyonread_adam_m_dense_21_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_11/DisableCopyOnReadDisableCopyOnRead0read_11_disablecopyonread_adam_v_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp0read_11_disablecopyonread_adam_v_dense_21_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_12/DisableCopyOnReadDisableCopyOnRead.read_12_disablecopyonread_adam_m_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp.read_12_disablecopyonread_adam_m_dense_21_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_13/DisableCopyOnReadDisableCopyOnRead.read_13_disablecopyonread_adam_v_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp.read_13_disablecopyonread_adam_v_dense_21_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnRead0read_14_disablecopyonread_adam_m_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp0read_14_disablecopyonread_adam_m_dense_22_kernel^Read_14/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_adam_v_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_adam_v_dense_22_kernel^Read_15/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_adam_m_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_adam_m_dense_22_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_17/DisableCopyOnReadDisableCopyOnRead.read_17_disablecopyonread_adam_v_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp.read_17_disablecopyonread_adam_v_dense_22_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnRead0read_18_disablecopyonread_adam_m_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp0read_18_disablecopyonread_adam_m_dense_23_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_v_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_v_dense_23_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_20/DisableCopyOnReadDisableCopyOnRead.read_20_disablecopyonread_adam_m_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp.read_20_disablecopyonread_adam_m_dense_23_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_adam_v_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_adam_v_dense_23_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead0read_22_disablecopyonread_adam_m_dense_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp0read_22_disablecopyonread_adam_m_dense_24_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_v_dense_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_v_dense_24_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_24/DisableCopyOnReadDisableCopyOnRead.read_24_disablecopyonread_adam_m_dense_24_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp.read_24_disablecopyonread_adam_m_dense_24_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_adam_v_dense_24_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_adam_v_dense_24_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp!read_26_disablecopyonread_total_2^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_count_2^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_28/DisableCopyOnReadDisableCopyOnRead!read_28_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp!read_28_disablecopyonread_total_1^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_29/DisableCopyOnReadDisableCopyOnRead!read_29_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp!read_29_disablecopyonread_count_1^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_30/DisableCopyOnReadDisableCopyOnReadread_30_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpread_30_disablecopyonread_total^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_31/DisableCopyOnReadDisableCopyOnReadread_31_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpread_31_disablecopyonread_count^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*�
value�B�!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 */
dtypes%
#2!	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_64Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_65IdentityIdentity_64:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_65Identity_65:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes
: 
�	
�
-__inference_sequential_5_layer_call_fn_846559

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_846391o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_dense_22_layer_call_and_return_conditional_losses_846661

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_23_layer_call_and_return_conditional_losses_846272

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_23_layer_call_and_return_conditional_losses_846681

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846295
dense_21_input"
dense_21_846239:	�
dense_21_846241:	�#
dense_22_846256:
��
dense_22_846258:	�"
dense_23_846273:	�
dense_23_846275:!
dense_24_846289:
dense_24_846291:
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCalldense_21_inputdense_21_846239dense_21_846241*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_846238�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_846256dense_22_846258*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_846255�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_846273dense_23_846275*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_846272�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_846289dense_24_846291*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_846288x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�	
�
D__inference_dense_24_layer_call_and_return_conditional_losses_846288

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�,
�
!__inference__wrapped_model_846223
dense_21_inputG
4sequential_5_dense_21_matmul_readvariableop_resource:	�D
5sequential_5_dense_21_biasadd_readvariableop_resource:	�H
4sequential_5_dense_22_matmul_readvariableop_resource:
��D
5sequential_5_dense_22_biasadd_readvariableop_resource:	�G
4sequential_5_dense_23_matmul_readvariableop_resource:	�C
5sequential_5_dense_23_biasadd_readvariableop_resource:F
4sequential_5_dense_24_matmul_readvariableop_resource:C
5sequential_5_dense_24_biasadd_readvariableop_resource:
identity��,sequential_5/dense_21/BiasAdd/ReadVariableOp�+sequential_5/dense_21/MatMul/ReadVariableOp�,sequential_5/dense_22/BiasAdd/ReadVariableOp�+sequential_5/dense_22/MatMul/ReadVariableOp�,sequential_5/dense_23/BiasAdd/ReadVariableOp�+sequential_5/dense_23/MatMul/ReadVariableOp�,sequential_5/dense_24/BiasAdd/ReadVariableOp�+sequential_5/dense_24/MatMul/ReadVariableOp�
+sequential_5/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_21_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_5/dense_21/MatMulMatMuldense_21_input3sequential_5/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_5/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_21/BiasAddBiasAdd&sequential_5/dense_21/MatMul:product:04sequential_5/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_5/dense_21/ReluRelu&sequential_5/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
+sequential_5/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_5/dense_22/MatMulMatMul(sequential_5/dense_21/Relu:activations:03sequential_5/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_5/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_5/dense_22/BiasAddBiasAdd&sequential_5/dense_22/MatMul:product:04sequential_5/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_5/dense_22/ReluRelu&sequential_5/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
+sequential_5/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_23_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_5/dense_23/MatMulMatMul(sequential_5/dense_22/Relu:activations:03sequential_5/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_5/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_5/dense_23/BiasAddBiasAdd&sequential_5/dense_23/MatMul:product:04sequential_5/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
sequential_5/dense_23/ReluRelu&sequential_5/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:����������
+sequential_5/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_24_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_5/dense_24/MatMulMatMul(sequential_5/dense_23/Relu:activations:03sequential_5/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_5/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_5/dense_24/BiasAddBiasAdd&sequential_5/dense_24/MatMul:product:04sequential_5/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
IdentityIdentity&sequential_5/dense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential_5/dense_21/BiasAdd/ReadVariableOp,^sequential_5/dense_21/MatMul/ReadVariableOp-^sequential_5/dense_22/BiasAdd/ReadVariableOp,^sequential_5/dense_22/MatMul/ReadVariableOp-^sequential_5/dense_23/BiasAdd/ReadVariableOp,^sequential_5/dense_23/MatMul/ReadVariableOp-^sequential_5/dense_24/BiasAdd/ReadVariableOp,^sequential_5/dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2\
,sequential_5/dense_21/BiasAdd/ReadVariableOp,sequential_5/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_21/MatMul/ReadVariableOp+sequential_5/dense_21/MatMul/ReadVariableOp2\
,sequential_5/dense_22/BiasAdd/ReadVariableOp,sequential_5/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_22/MatMul/ReadVariableOp+sequential_5/dense_22/MatMul/ReadVariableOp2\
,sequential_5/dense_23/BiasAdd/ReadVariableOp,sequential_5/dense_23/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_23/MatMul/ReadVariableOp+sequential_5/dense_23/MatMul/ReadVariableOp2\
,sequential_5/dense_24/BiasAdd/ReadVariableOp,sequential_5/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_24/MatMul/ReadVariableOp+sequential_5/dense_24/MatMul/ReadVariableOp:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846391

inputs"
dense_21_846370:	�
dense_21_846372:	�#
dense_22_846375:
��
dense_22_846377:	�"
dense_23_846380:	�
dense_23_846382:!
dense_24_846385:
dense_24_846387:
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinputsdense_21_846370dense_21_846372*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_846238�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_846375dense_22_846377*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_846255�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_846380dense_23_846382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_846272�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_846385dense_24_846387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_846288x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_dense_21_layer_call_fn_846630

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_846238p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846621

inputs:
'dense_21_matmul_readvariableop_resource:	�7
(dense_21_biasadd_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�:
'dense_23_matmul_readvariableop_resource:	�6
(dense_23_biasadd_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource:6
(dense_24_biasadd_readvariableop_resource:
identity��dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0|
dense_21/MatMulMatMulinputs&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846590

inputs:
'dense_21_matmul_readvariableop_resource:	�7
(dense_21_biasadd_readvariableop_resource:	�;
'dense_22_matmul_readvariableop_resource:
��7
(dense_22_biasadd_readvariableop_resource:	�:
'dense_23_matmul_readvariableop_resource:	�6
(dense_23_biasadd_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource:6
(dense_24_biasadd_readvariableop_resource:
identity��dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0|
dense_21/MatMulMatMulinputs&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_dense_21_layer_call_and_return_conditional_losses_846238

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
D__inference_dense_24_layer_call_and_return_conditional_losses_846700

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_dense_24_layer_call_fn_846690

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_846288o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_dense_22_layer_call_fn_846650

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_846255p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846319
dense_21_input"
dense_21_846298:	�
dense_21_846300:	�#
dense_22_846303:
��
dense_22_846305:	�"
dense_23_846308:	�
dense_23_846310:!
dense_24_846313:
dense_24_846315:
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCalldense_21_inputdense_21_846298dense_21_846300*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_846238�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_846303dense_22_846305*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_846255�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_846308dense_23_846310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_846272�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_846313dense_24_846315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_846288x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�
�
)__inference_dense_23_layer_call_fn_846670

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_846272o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
dense_21_input7
 serving_default_dense_21_input:0���������<
dense_240
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
3trace_0
4trace_1
5trace_2
6trace_32�
-__inference_sequential_5_layer_call_fn_846365
-__inference_sequential_5_layer_call_fn_846410
-__inference_sequential_5_layer_call_fn_846538
-__inference_sequential_5_layer_call_fn_846559�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3trace_0z4trace_1z5trace_2z6trace_3
�
7trace_0
8trace_1
9trace_2
:trace_32�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846295
H__inference_sequential_5_layer_call_and_return_conditional_losses_846319
H__inference_sequential_5_layer_call_and_return_conditional_losses_846590
H__inference_sequential_5_layer_call_and_return_conditional_losses_846621�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z7trace_0z8trace_1z9trace_2z:trace_3
�B�
!__inference__wrapped_model_846223dense_21_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
;
_variables
<_iterations
=_learning_rate
>_index_dict
?
_momentums
@_velocities
A_update_step_xla"
experimentalOptimizer
,
Bserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Htrace_02�
)__inference_dense_21_layer_call_fn_846630�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
�
Itrace_02�
D__inference_dense_21_layer_call_and_return_conditional_losses_846641�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zItrace_0
": 	�2dense_21/kernel
:�2dense_21/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Otrace_02�
)__inference_dense_22_layer_call_fn_846650�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0
�
Ptrace_02�
D__inference_dense_22_layer_call_and_return_conditional_losses_846661�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zPtrace_0
#:!
��2dense_22/kernel
:�2dense_22/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Vtrace_02�
)__inference_dense_23_layer_call_fn_846670�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zVtrace_0
�
Wtrace_02�
D__inference_dense_23_layer_call_and_return_conditional_losses_846681�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zWtrace_0
": 	�2dense_23/kernel
:2dense_23/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�
]trace_02�
)__inference_dense_24_layer_call_fn_846690�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z]trace_0
�
^trace_02�
D__inference_dense_24_layer_call_and_return_conditional_losses_846700�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z^trace_0
!:2dense_24/kernel
:2dense_24/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
5
_0
`1
a2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_5_layer_call_fn_846365dense_21_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_5_layer_call_fn_846410dense_21_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_5_layer_call_fn_846538inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_5_layer_call_fn_846559inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846295dense_21_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846319dense_21_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846590inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_5_layer_call_and_return_conditional_losses_846621inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
<0
b1
c2
d3
e4
f5
g6
h7
i8
j9
k10
l11
m12
n13
o14
p15
q16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
X
b0
d1
f2
h3
j4
l5
n6
p7"
trackable_list_wrapper
X
c0
e1
g2
i3
k4
m5
o6
q7"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_846517dense_21_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_dense_21_layer_call_fn_846630inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_21_layer_call_and_return_conditional_losses_846641inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_dense_22_layer_call_fn_846650inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_22_layer_call_and_return_conditional_losses_846661inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_dense_23_layer_call_fn_846670inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_23_layer_call_and_return_conditional_losses_846681inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_dense_24_layer_call_fn_846690inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_24_layer_call_and_return_conditional_losses_846700inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
r	variables
s	keras_api
	ttotal
	ucount"
_tf_keras_metric
^
v	variables
w	keras_api
	xtotal
	ycount
z
_fn_kwargs"
_tf_keras_metric
^
{	variables
|	keras_api
	}total
	~count

_fn_kwargs"
_tf_keras_metric
':%	�2Adam/m/dense_21/kernel
':%	�2Adam/v/dense_21/kernel
!:�2Adam/m/dense_21/bias
!:�2Adam/v/dense_21/bias
(:&
��2Adam/m/dense_22/kernel
(:&
��2Adam/v/dense_22/kernel
!:�2Adam/m/dense_22/bias
!:�2Adam/v/dense_22/bias
':%	�2Adam/m/dense_23/kernel
':%	�2Adam/v/dense_23/kernel
 :2Adam/m/dense_23/bias
 :2Adam/v/dense_23/bias
&:$2Adam/m/dense_24/kernel
&:$2Adam/v/dense_24/kernel
 :2Adam/m/dense_24/bias
 :2Adam/v/dense_24/bias
.
t0
u1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
:  (2total
:  (2count
.
x0
y1"
trackable_list_wrapper
-
v	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_846223x$%,-7�4
-�*
(�%
dense_21_input���������
� "3�0
.
dense_24"�
dense_24����������
D__inference_dense_21_layer_call_and_return_conditional_losses_846641d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_21_layer_call_fn_846630Y/�,
%�"
 �
inputs���������
� ""�
unknown�����������
D__inference_dense_22_layer_call_and_return_conditional_losses_846661e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_22_layer_call_fn_846650Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
D__inference_dense_23_layer_call_and_return_conditional_losses_846681d$%0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_23_layer_call_fn_846670Y$%0�-
&�#
!�
inputs����������
� "!�
unknown����������
D__inference_dense_24_layer_call_and_return_conditional_losses_846700c,-/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_24_layer_call_fn_846690X,-/�,
%�"
 �
inputs���������
� "!�
unknown����������
H__inference_sequential_5_layer_call_and_return_conditional_losses_846295y$%,-?�<
5�2
(�%
dense_21_input���������
p

 
� ",�)
"�
tensor_0���������
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_846319y$%,-?�<
5�2
(�%
dense_21_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_846590q$%,-7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_846621q$%,-7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
-__inference_sequential_5_layer_call_fn_846365n$%,-?�<
5�2
(�%
dense_21_input���������
p

 
� "!�
unknown����������
-__inference_sequential_5_layer_call_fn_846410n$%,-?�<
5�2
(�%
dense_21_input���������
p 

 
� "!�
unknown����������
-__inference_sequential_5_layer_call_fn_846538f$%,-7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
-__inference_sequential_5_layer_call_fn_846559f$%,-7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
$__inference_signature_wrapper_846517�$%,-I�F
� 
?�<
:
dense_21_input(�%
dense_21_input���������"3�0
.
dense_24"�
dense_24���������