mailer_domain = "lab@cmdm.csie.ntu.edu.tw"

class UserMailer < Devise::Mailer
    def confirmation_instructions(record, token, opts={})
        headers["Custom-header"] = "Bar"
        opts[:subject] = 'Thank for your registration for FastEval Parkinsonism - account confirmation'
        opts[:from] = mailer_domain
        opts[:reply_to] = mailer_domain
        super
    end

    def reset_password_instructions(record, token, opts={})
        headers["Custom-header"] = "Bar"
        opts[:subject] = 'Thank for your using FastEval Parkinsonism - password reset/change'
        opts[:from] = mailer_domain
        opts[:reply_to] = mailer_domain
        super
    end
end
