# pccm.ParameterizedClass 

* Excerpted from the [Introduction of PCCM](https://github.com/FindDefinition/PCCM/tree/master/docs) by its author.

ParameterizedClass is the most significant part of PCCM. 

Recall C++ template classes, we can add  type/non-type to template arguments of template class, do some stuffs on these type/non-type,  then save result type/non-type by assign a new alias or static const inside the class body. 

This is called template meta programming. We can use meta programming to convert types or do computation in compile time. 

However, there are lots of limitations of C++ template meta programming. 

Firstly, we cannot write complex code such as if/for/while to handle types. Secondly, we cannot add error handling for meta program. Thirdly, we cannot declare meta-type of type arguments, so the intelligence engine cannot inference results of meta program. 

PCCM is a new framework to handle the problem of C++ template meta programming.



#### The pccm.ParamClass inherit pccm.Class. 

#### But have different code generation logic. 

A most important difference is that pccm.ParamClass can accept arguments in \_\_init\_\_, implying that a pccm.ParamClass can be parameterized like a C++ template class. 

This feature make  pccm.ParamClass not unique in a project, it’s unique in instance level, not type level of pccm.Class.  

PCCM instantiate pccm.ParamClass instance to namespace in current pccm.Class/ParamClass by  generate code in an inner namespace. 



