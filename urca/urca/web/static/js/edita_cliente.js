"use strict";

function voltar () {
    window.location.href = "/lista_clientes";
};

// Class Definition
var KTEditaCliente = function () {
	var _buttonSpinnerClasses = 'spinner spinner-right spinner-white pr-15';

	var _handleIncluirCliente = function () {
		var form = KTUtil.getById('kt_incluir_cliente_form');
		var formSubmitUrl = KTUtil.attr(form, 'action');
		var formSubmitButton = KTUtil.getById('kt_incluir_cliente_submit');

		if (!form) {
			return;
		}

		FormValidation
			.formValidation(
				form,
				{
					fields: {
						str_nome: {
							validators: {
								notEmpty: {
									message: 'Nome é requerido!'
								}
							}
						},
                        str_email_admin: {
                            validators: {
                                notEmpty: {
                                    message: 'Eamil do admin é requerido!'
                                }
                            }
                        },
                        str_token_teste: {
                            validators: {
                                notEmpty: {
                                    message: 'Token é requerido!'
                                }
                            }
                        },
                        int_credito_sms: {
                            validators: {
                                notEmpty: {
                                    message: 'Saldo SMS é requerido!'
                                },
                                between: {
                                    min: 0,
                                    max: 9999999,
                                    message: 'Campo saldo sms precisar estar entre 0 a 9999999'
                                }
                            }
                        },
                        int_credito_email: {
                            validators: {
                                notEmpty: {
                                    message: 'Saldo Email é requerido!'
                                },
                                between: {
                                    min: 0,
                                    max: 9999999,
                                    message: 'Campo saldo email precisar estar entre 0 a 9999999'
                                }
                            }
                        },
                        int_limite_eleicao: {
                            validators: {
                                notEmpty: {
                                    message: 'Máximo de eleições é requerido!'
                                },
                                between: {
                                    min: 0,
                                    max: 9999999,
                                    message: 'Campo Máximo de eleições precisar estar entre 0 a 9999999'
                                }
                            }
                        }
					},
					plugins: {
						trigger: new FormValidation.plugins.Trigger(),
						submitButton: new FormValidation.plugins.SubmitButton(),
						//defaultSubmit: new FormValidation.plugins.DefaultSubmit(), // Uncomment this line to enable normal button submit after form validation
						bootstrap: new FormValidation.plugins.Bootstrap({
							//	eleInvalidClass: '', // Repace with uncomment to hide bootstrap validation icons
							//	eleValidClass: '',   // Repace with uncomment to hide bootstrap validation icons
						})
					}
				}
			)
			.on('core.form.valid', function () {
				// Show loading state on button
				KTUtil.btnWait(formSubmitButton, _buttonSpinnerClasses, "Aguarde");

				// Simulate Ajax request
				setTimeout(function () {
					KTUtil.btnRelease(formSubmitButton);
                }, 2000);
                
                var str_nome=document.getElementById('str_nome').value;
                var str_email_admin=document.getElementById('str_email_admin').value;
                var str_token_teste=document.getElementById('str_token_teste').value;
                var int_credito_sms=document.getElementById('int_credito_sms').value;
                var int_credito_email=document.getElementById('int_credito_email').value;
                var int_limite_eleicao=document.getElementById('int_limite_eleicao').value;
                var bol_status=document.getElementById('bol_status').checked;


                if( bol_status ) {
                    bol_status = 't';
                }else{
                    bol_status = 'f';
                }


                var op=document.getElementById('op').value;
                
                if( op === "incluir" ) {
                    
                    $.post( "/_inclui_cliente", {str_nome: str_nome, str_email_admin: str_email_admin, str_token_teste:str_token_teste, int_credito_sms:int_credito_sms,int_credito_email:int_credito_email,int_limite_eleicao:int_limite_eleicao, bol_status:bol_status})
                    .done(function(data){
                        var res = JSON.parse(data);
                        // alert(res.msg);
                        if (res.status == '1') {
                            
                            Swal.fire({
                                text: res.msg,
                                icon: "error",
                                buttonsStyling: false,
                                confirmButtonText: "Ok!",
                                customClass: {
                                    confirmButton: "btn font-weight-bold btn-light-primary"
                                }
                            }).then(function () {
                                KTUtil.scrollTop();
                            });

                        } else {
                            
                            Swal.fire({
                                text: "Cliente incluido com sucesso!",
                                icon: "success",
                                buttonsStyling: false,
                                confirmButtonText: "Ok!",
                                customClass: {
                                    confirmButton: "btn font-weight-bold btn-light-primary"
                                }
                            }).then(function () {
            
                                KTUtil.scrollTop();
                                window.location.href = "/lista_clientes";  

                            });
                            
                            
                        }
                    });

                }else{
                    console.log(str_token_teste);
                    $.post( "/_altera_cliente", {str_nome: str_nome, str_email_admin:str_email_admin, str_token_teste:str_token_teste, int_credito_sms:int_credito_sms,int_credito_email:int_credito_email,int_limite_eleicao:int_limite_eleicao, bol_status:bol_status})
                        .done(function(data){
                            var res = JSON.parse(data);
                            // alert(res.msg);
                            if (res.status == '1') {
                                
                                Swal.fire({
                                    text: res.msg,
                                    icon: "error",
                                    buttonsStyling: false,
                                    confirmButtonText: "Ok!",
                                    customClass: {
                                        confirmButton: "btn font-weight-bold btn-light-primary"
                                    }
                                }).then(function () {
                                    KTUtil.scrollTop();
                                });

                            } else {

                                Swal.fire({
                                    text: "Cliente alterado com sucesso!",
                                    icon: "success",
                                    buttonsStyling: false,
                                    confirmButtonText: "Ok!",
                                    customClass: {
                                        confirmButton: "btn font-weight-bold btn-light-primary"
                                    }
                                }).then(function () {
                
                                    KTUtil.scrollTop();
                                    window.location.href = "/lista_clientes";  
    
                                });

                            }
                            // window.location.href = "/abre_relat";
                            //window.location.href="/login" ;
                        });

                }

			})
			.on('core.form.invalid', function () {
				Swal.fire({
					text: "Desculpe, alguns erros foram detectados no formulário, por favor preencher novamente!",
					icon: "error",
					buttonsStyling: false,
					confirmButtonText: "Ok!",
					customClass: {
						confirmButton: "btn font-weight-bold btn-light-primary"
					}
				}).then(function () {
					KTUtil.scrollTop();
				});
			});
	}

	// Public Functions
	return {
		init: function () {
			_handleIncluirCliente();
		}
	};
}();

// Class Initialization
jQuery(document).ready(function () {
	KTEditaCliente.init();
});
