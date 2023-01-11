class UserMailer < Devise::Mailer
    def confirmation_instructions(record, token, opts={})
        headers["Custom-header"] = "Bar"
        opts[:subject] = 'Thank for your registration for FastEval Parkinsonism - account confirmation'
        opts[:from] = 'do-not-reply@fasteval_parkinsonism.cmdm.tw'
        opts[:reply_to] = 'do-not-reply@fasteval_parkinsonism.cmdm.tw'
        super
    end

    def reset_password_instructions(record, token, opts={})
        headers["Custom-header"] = "Bar"
        opts[:subject] = 'Thank for your using FastEval Parkinsonism - password reset/change'
        opts[:from] = 'do-not-reply@fasteval_parkinsonism.cmdm.tw'
        opts[:reply_to] = 'do-not-reply@fasteval_parkinsonism.cmdm.tw'
        super
    end
end
