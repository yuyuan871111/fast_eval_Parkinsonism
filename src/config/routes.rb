Rails.application.routes.draw do
  mount RailsAdmin::Engine => '/admin', as: 'rails_admin'

  devise_for :users, controllers: { 
    sessions: 'users/sessions', 
    registrations: 'users/registrations',
    passwords: 'users/passwords',
    confirmations: 'users/confirmations' 
  }

  devise_scope :user do  
    post '/users/sign_in' => 'users/sessions#new'
    get '/users/sign_out' => 'users/sessions#destroy'
    get '/users/sign_up' => 'users/registrations#new'
  end

  # preview at http://localhost:3000/letter_opener
  if Rails.env.development?
    mount LetterOpenerWeb::Engine, at: "/letter_opener"
  end
  # get 'home/index'
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Defines the root path route ("/")
  root "home#index", :as => 'root'
  get '/documents' => 'home#documents', :as => 'documents'
  get '/demo' => 'home#demo', :as => 'demo'
  get '/status' => 'home#status', :as => 'status'
  
  resources :uploads

  #get '/dashboard' => 'dashboard#index', :as => 'dashboard'
  resources :dashboard
  #get '/dashboard' => 'dashboard#download', :as => 'dashboard_download'
    
end
