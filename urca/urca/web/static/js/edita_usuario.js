"use strict";

function voltar () {
    window.location.href = "/lista_usuarios";
};

// Class Definition
var KTEditaCliente = function () {
	var _buttonSpinnerClasses = 'spinner spinner-right spinner-white pr-15';

	var _handleIncluirCliente = function () {
		var form = KTUtil.getById('kt_incluir_usuario_form');
		var formSubmitUrl = KTUtil.attr(form, 'action');
		var formSubmitButton = KTUtil.getById('kt_incluir_usuario_submit');

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
						str_email: {
							validators: {
								notEmpty: {
									message: 'Email é requerido!'
                                },
                                emailAddress: {
                                    message: 'Email inválido'
                                }
							}
                        },
                        
                        
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
                var str_telefone=document.getElementById('str_telefone').value;
                var str_email=document.getElementById('str_email').value;
                var str_senha=document.getElementById('str_senha').value;
                var fk_id_permissao=document.getElementById('fk_id_permissao').value;
                var fk_id_cliente=document.getElementById('fk_id_cliente').value;
                var bol_status=document.getElementById('bol_status').checked;
                var bol_admin=document.getElementById('bol_admin').checked;

                if( bol_status ) {
                    bol_status = 't';
                }else{
                    bol_status = 'f';
                }

                if( bol_admin ) {
                    bol_admin = 't';
                }else{
                    bol_admin = 'f';
                }



                var op=document.getElementById('op').value;
                
                if( op === "incluir" ) {
                    
                    $.post( "/_inclui_usuario", {str_nome: str_nome, str_telefone:str_telefone, str_email: str_email, str_senha:str_senha, fk_id_permissao:fk_id_permissao,fk_id_cliente:fk_id_cliente,bol_status:bol_status,bol_admin:bol_admin})
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
                                text: "Usuário incluido com sucesso!",
                                icon: "success",
                                buttonsStyling: false,
                                confirmButtonText: "Ok!",
                                customClass: {
                                    confirmButton: "btn font-weight-bold btn-light-primary"
                                }
                            }).then(function () {
            
                                KTUtil.scrollTop();
                                window.location.href = "/lista_usuarios";  

                            });
                            
                        }
                    });

                }else{
                    
                    $.post( "/_altera_usuario", {str_nome: str_nome, str_telefone:str_telefone, str_email: str_email,str_senha:str_senha, fk_id_permissao:fk_id_permissao,fk_id_cliente:fk_id_cliente,bol_status:bol_status,bol_admin:bol_admin})
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
                                    text: "Usuário alterado com sucesso!",
                                    icon: "success",
                                    buttonsStyling: false,
                                    confirmButtonText: "Ok!",
                                    customClass: {
                                        confirmButton: "btn font-weight-bold btn-light-primary"
                                    }
                                }).then(function () {
                
                                    KTUtil.scrollTop();
                                    window.location.href = "/lista_usuarios";  
    
                                });

                            }
                            
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
