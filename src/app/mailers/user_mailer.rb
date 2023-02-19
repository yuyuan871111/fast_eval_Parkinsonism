class UserMailer < Devise::Mailer
    def confirmation_instructions(record, token, opts={})
        headers["Custom-header"] = "Bar"
        opts[:subject] = 'Thank for your registration for FastEval Parkinsonism - account confirmation'
        opts[:from] = "lab@cmdm.csie.ntu.edu.tw"
        opts[:reply_to] = "lab@cmdm.csie.ntu.edu.tw"
        super
    end

    def reset_password_instructions(record, token, opts={})
        headers["Custom-header"] = "Bar"
        opts[:subject] = 'Thank for your using FastEval Parkinsonism - password reset/change'
        opts[:from] = "lab@cmdm.csie.ntu.edu.tw"
        opts[:reply_to] = "lab@cmdm.csie.ntu.edu.tw"
        super
    end
end
