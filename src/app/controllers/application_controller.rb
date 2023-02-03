class ApplicationController < ActionController::Base
  before_action :configure_permitted_parameters, if: :devise_controller?

  protected

  def configure_permitted_parameters
    devise_parameter_sanitizer.permit(:sign_up, keys: [:email])
    devise_parameter_sanitizer.permit(:account_update, keys: [:email])
  end
  private
    def authenticate_user!(opts={})
      if user_signed_in?
        super
      else
        redirect_to '/users/sign_in', notice: "Please Login to view that page!"
      end
    end
end
